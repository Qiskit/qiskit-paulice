// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! A CPU scoring engine, written against `include/paulice_engine.h` the way
//! an external engine is: it declares the ABI structs itself and computes
//! every result from the flat arrays it is handed. qiskit-paulice's own code
//! is used only for the Philox streams, which the ABI fixes.
//!
//! Registering it lets `run_conformance` drive the dispatch code end to end
//! without a GPU. Each test file is its own process, so each can register
//! once.

#![allow(dead_code)]

use _internal_r::philox::GeneratorStream;
use std::os::raw::{c_char, c_int, c_void};
use std::sync::atomic::{AtomicU8, Ordering};

const ABI_VERSION: u32 = 1;
const LER_MAX_WORDS: i32 = 1024;
const LER_MAX_GENERATORS: u32 = 65535;
const LER_DRAW_OVERFLOW: u32 = u32::MAX;
const MAX_QUBITS: i32 = 256;

const GATE_CNOT: u32 = 0;
const GATE_CZ: u32 = 1;
const GATE_H: u32 = 2;
const GATE_S: u32 = 3;
const GATE_SD: u32 = 4;
const GATE_SQRTX: u32 = 5;
const GATE_SQRTXD: u32 = 6;

const NOISE_DEPOLARIZING: i32 = 0;

#[repr(C)]
pub struct NoiseBlock {
    kind: i32,
    rate: f64,
}

#[repr(C)]
pub struct GammaBatch {
    n_candidates: i32,
    nqubits: i32,
    gates: *const u32,
    gate_offset: *const i32,
    n_two_qubit: *const i32,
    post_seed: *const u64,
    post_row_offset: *const i32,
    logical_seed: *const u64,
    logical_row_offset: *const i32,
    row_words: i32,
    readout_column: *const i32,
    blocks: *const NoiseBlock,
    n_blocks: i32,
    out_rates: *mut f64,
}

#[repr(C)]
pub struct LerBatch {
    gen_words: *const u64,
    n_generators: i32,
    stride: i32,
    planes_end: i32,
    post_end: i32,
    offsets: *const u32,
    indices: *const u32,
    n_shots: i32,
    n_indices: i32,
    out_accepted: *mut u64,
    out_logical: *mut u64,
}

#[repr(C)]
pub struct LerDrawParity {
    seed: u64,
    n_generators: u32,
    nshots: u32,
    cdf_all: *const f64,
    n_cdf: u32,
    table_offset: *const u32,
    table_len: *const u32,
    table_klo: *const u32,
    n_tables: u32,
    table_of: *const u32,
    probe_words: u32,
    out_words: *mut u64,
    out_counts: *mut u32,
    out_pos: *mut u32,
    pos_stride: u32,
    out_after: *mut u64,
}

#[repr(C)]
pub struct LerTranspose {
    pos: *const u32,
    pos_stride: u32,
    gen_counts: *const u32,
    n_generators: u32,
    n_shots: u32,
    out_offsets: *mut u32,
    out_indices: *mut u32,
    n_indices_capacity: u32,
    out_nonempty: *mut u64,
}

#[repr(C)]
pub struct LerSampleScore {
    gen_words: *const u64,
    n_generators: i32,
    stride: i32,
    planes_end: i32,
    post_end: i32,
    n_shots: i32,
    seed: u64,
    cdf_all: *const f64,
    n_cdf: u32,
    table_offset: *const u32,
    table_len: *const u32,
    table_klo: *const u32,
    n_tables: u32,
    table_of: *const u32,
    pos_stride: u32,
    reuse_sample: i32,
    out_accepted: *mut u64,
    out_logical: *mut u64,
    out_nonempty: *mut u64,
}

type Ctx = c_void;

#[repr(C)]
pub struct Engine {
    abi_version: u32,
    size: u32,
    name: *const c_char,
    device_count: unsafe extern "C" fn() -> c_int,
    create: unsafe extern "C" fn(c_int) -> *mut Ctx,
    destroy: unsafe extern "C" fn(*mut Ctx),
    gamma_batch: unsafe extern "C" fn(*mut Ctx, *const GammaBatch) -> c_int,
    ler_sample_score: unsafe extern "C" fn(*mut Ctx, *const LerSampleScore) -> c_int,
    ler_score: unsafe extern "C" fn(*mut Ctx, *const LerBatch) -> c_int,
    ler_draw_parity: unsafe extern "C" fn(*mut Ctx, *const LerDrawParity) -> c_int,
    ler_transpose: unsafe extern "C" fn(*mut Ctx, *const LerTranspose) -> c_int,
}

// The table is immutable and its entries are plain functions.
unsafe impl Sync for Engine {}

/// A deliberate bug to inject, so a test can check that conformance notices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Fault {
    None = 0,
    /// Gamma counts one generator too many in the first noise block.
    GammaCount = 1,
    /// Scoring reports one logical error too many.
    LogicalCount = 2,
    /// The draw keeps duplicate shot positions.
    DrawDuplicates = 3,
}

static FAULT: AtomicU8 = AtomicU8::new(Fault::None as u8);

pub fn inject(fault: Fault) {
    FAULT.store(fault as u8, Ordering::Relaxed);
}

fn faulty(fault: Fault) -> bool {
    FAULT.load(Ordering::Relaxed) == fault as u8
}

/// The reference engine's table.
pub static REFERENCE: Engine = Engine {
    abi_version: ABI_VERSION,
    size: std::mem::size_of::<Engine>() as u32,
    name: c"reference".as_ptr(),
    device_count,
    create,
    destroy,
    gamma_batch,
    ler_sample_score,
    ler_score,
    ler_draw_parity,
    ler_transpose,
};

/// The sample left by the last successful draw, which `reuse_sample` scores.
struct Session {
    resident: Option<Resident>,
}

struct Resident {
    shots: u32,
    generators: u32,
    offsets: Vec<u32>,
    indices: Vec<u32>,
}

unsafe fn slice<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if len == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}

unsafe fn slice_mut<'a, T>(ptr: *mut T, len: usize) -> &'a mut [T] {
    if len == 0 {
        &mut []
    } else {
        unsafe { std::slice::from_raw_parts_mut(ptr, len) }
    }
}

unsafe fn session<'a>(ctx: *mut Ctx) -> &'a mut Session {
    unsafe { &mut *ctx.cast::<Session>() }
}

unsafe extern "C" fn device_count() -> c_int {
    1
}

unsafe extern "C" fn create(device_id: c_int) -> *mut Ctx {
    if device_id != 0 {
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(Session { resident: None })).cast()
}

unsafe extern "C" fn destroy(ctx: *mut Ctx) {
    if !ctx.is_null() {
        drop(unsafe { Box::from_raw(ctx.cast::<Session>()) });
    }
}

// ---------------------------------------------------------------- gamma

/// One cumulant row: the x and z bits of a Pauli over the qubits.
#[derive(Clone)]
struct Row {
    x: Vec<u64>,
    z: Vec<u64>,
}

fn bit(words: &[u64], q: usize) -> bool {
    (words[q / 64] >> (q % 64)) & 1 == 1
}

fn put(words: &mut [u64], q: usize, value: bool) {
    let mask = 1u64 << (q % 64);
    if value {
        words[q / 64] |= mask;
    } else {
        words[q / 64] &= !mask;
    }
}

/// Phase-free symplectic action of a gate. Every gate is its own inverse
/// here, so the backward walk uses the same rule.
fn apply(row: &mut Row, packed: u32) {
    let op = packed >> 24;
    let a = ((packed >> 12) & 0xfff) as usize;
    let b = (packed & 0xfff) as usize;
    match op {
        GATE_H => {
            let (xa, za) = (bit(&row.x, a), bit(&row.z, a));
            put(&mut row.x, a, za);
            put(&mut row.z, a, xa);
        }
        GATE_S | GATE_SD => {
            let v = bit(&row.z, a) ^ bit(&row.x, a);
            put(&mut row.z, a, v);
        }
        GATE_SQRTX | GATE_SQRTXD => {
            let v = bit(&row.x, a) ^ bit(&row.z, a);
            put(&mut row.x, a, v);
        }
        GATE_CZ => {
            let zb = bit(&row.z, b) ^ bit(&row.x, a);
            let za = bit(&row.z, a) ^ bit(&row.x, b);
            put(&mut row.z, b, zb);
            put(&mut row.z, a, za);
        }
        GATE_CNOT => {
            let xb = bit(&row.x, b) ^ bit(&row.x, a);
            let za = bit(&row.z, a) ^ bit(&row.z, b);
            put(&mut row.x, b, xb);
            put(&mut row.z, a, za);
        }
        _ => {}
    }
}

fn is_two_qubit(packed: u32) -> bool {
    matches!(packed >> 24, GATE_CZ | GATE_CNOT)
}

/// Walks each seed row backwards through `gates` and returns, per row, its
/// `(x, z)` bit at every column: the two wires of each 2-qubit gate from the
/// last gate to the first, then each input wire.
fn propagate(
    gates: &[u32],
    seeds: &[u64],
    row_words: usize,
    nqubits: usize,
) -> Vec<Vec<(bool, bool)>> {
    // Rows without words are the identity, which anticommutes with nothing.
    if row_words == 0 {
        return Vec::new();
    }
    let words = nqubits.div_ceil(64).max(row_words);
    seeds
        .chunks_exact(row_words)
        .map(|seed| {
            let mut row = Row {
                x: vec![0; words],
                z: vec![0; words],
            };
            row.z[..row_words].copy_from_slice(seed);
            let mut columns = Vec::new();
            for &packed in gates.iter().rev() {
                if is_two_qubit(packed) {
                    let a = ((packed >> 12) & 0xfff) as usize;
                    let b = (packed & 0xfff) as usize;
                    columns.push((bit(&row.x, a), bit(&row.z, a)));
                    columns.push((bit(&row.x, b), bit(&row.z, b)));
                }
                apply(&mut row, packed);
            }
            for q in 0..nqubits {
                columns.push((bit(&row.x, q), bit(&row.z, q)));
            }
            columns
        })
        .collect()
}

/// Whether any row anticommutes with the generator `terms`, a list of
/// `(column, pauli)` with `pauli` 1, 2, 3 for X, Y, Z.
fn covered(table: &[Vec<(bool, bool)>], terms: &[(i32, u8)]) -> bool {
    table.iter().any(|columns| {
        terms.iter().fold(false, |acc, &(col, p)| {
            if col < 0 {
                return acc;
            }
            let (x, z) = columns[col as usize];
            let xe = p == 1 || p == 2;
            let ze = p == 2 || p == 3;
            acc ^ (ze && x) ^ (xe && z)
        })
    })
}

unsafe extern "C" fn gamma_batch(ctx: *mut Ctx, batch: *const GammaBatch) -> c_int {
    if ctx.is_null() || batch.is_null() {
        return -1;
    }
    let b = unsafe { &*batch };
    if b.n_candidates <= 0 || b.n_blocks <= 0 || b.nqubits <= 0 || b.nqubits > MAX_QUBITS {
        return -1;
    }
    let n = b.n_candidates as usize;
    let nqubits = b.nqubits as usize;
    let row_words = b.row_words as usize;
    let gate_offset = unsafe { slice(b.gate_offset, n + 1) };
    let post_offset = unsafe { slice(b.post_row_offset, n + 1) };
    let logical_offset = unsafe { slice(b.logical_row_offset, n + 1) };
    let gates = unsafe { slice(b.gates, gate_offset[n] as usize) };
    let n_two_qubit = unsafe { slice(b.n_two_qubit, n) };
    let post_seed = unsafe { slice(b.post_seed, post_offset[n] as usize * row_words) };
    let logical_seed = unsafe { slice(b.logical_seed, logical_offset[n] as usize * row_words) };
    let readout = unsafe { slice(b.readout_column, n * nqubits) };
    let blocks = unsafe { slice(b.blocks, b.n_blocks as usize) };
    let out = unsafe { slice_mut(b.out_rates, n) };

    let rows = |offsets: &[i32], seeds: &'_ [u64], c: usize| -> Vec<u64> {
        seeds[offsets[c] as usize * row_words..offsets[c + 1] as usize * row_words].to_vec()
    };
    for c in 0..n {
        let circuit = &gates[gate_offset[c] as usize..gate_offset[c + 1] as usize];
        let n2q = n_two_qubit[c] as usize;
        let post = propagate(
            circuit,
            &rows(post_offset, post_seed, c),
            row_words,
            nqubits,
        );
        let logical = propagate(
            circuit,
            &rows(logical_offset, logical_seed, c),
            row_words,
            nqubits,
        );
        let readout = &readout[c * nqubits..(c + 1) * nqubits];

        let mut acc = 0.0f64;
        for (i, block) in blocks.iter().enumerate() {
            let generators: Vec<Vec<(i32, u8)>> = if block.kind == NOISE_DEPOLARIZING {
                // 15 per 2-qubit gate in ascending gate order; pairs row-major
                // over (p0, p1) without the identity. The walk visits the gates
                // last first, so gate `rank` owns columns 2 * (n2q - 1 - rank).
                (0..15 * n2q)
                    .map(|g| {
                        let col0 = (2 * (n2q - 1 - g / 15)) as i32;
                        let pair = (g % 15 + 1) as u8;
                        let (p0, p1) = (pair >> 2, pair & 3);
                        let mut terms = Vec::new();
                        if p0 != 0 {
                            terms.push((col0, p0));
                        }
                        if p1 != 0 {
                            terms.push((col0 + 1, p1));
                        }
                        terms
                    })
                    .collect()
            } else {
                readout.iter().map(|&col| vec![(col, 1u8)]).collect()
            };
            let mut count = generators
                .iter()
                .filter(|terms| !covered(&post, terms) && covered(&logical, terms))
                .count();
            if i == 0 && faulty(Fault::GammaCount) {
                count += 1;
            }
            for _ in 0..count {
                acc += block.rate;
            }
        }
        out[c] = acc;
    }
    0
}

// ------------------------------------------------------------------ LER

struct Tables<'a> {
    cdf: &'a [f64],
    offset: &'a [u32],
    len: &'a [u32],
    klo: &'a [u32],
    of: &'a [u32],
}

impl Tables<'_> {
    /// The window's low count plus the number of CDF entries at or below `u`.
    fn sample(&self, generator: usize, u: f64) -> u32 {
        let t = self.of[generator] as usize;
        let start = self.offset[t] as usize;
        let cdf = &self.cdf[start..start + self.len[t] as usize];
        self.klo[t] + cdf.partition_point(|&v| v <= u) as u32
    }
}

/// One generator's distinct shot positions in draw order, or `None` if its
/// count exceeds `capacity`. Batches are sized to the deficit, as on the host.
fn draw(
    seed: u64,
    generator: usize,
    tables: &Tables,
    nshots: u32,
    capacity: u32,
) -> Option<(Vec<u32>, GeneratorStream)> {
    let mut stream = GeneratorStream::new(seed, generator);
    let want = tables.sample(generator, stream.next_f64()).min(nshots);
    if want > capacity {
        return None;
    }
    let mut seen = vec![false; nshots as usize];
    let mut out = Vec::with_capacity(want as usize);
    while (out.len() as u32) < want {
        let deficit = want - out.len() as u32;
        for _ in 0..deficit {
            let p = stream.next_below(nshots as u64) as u32;
            if !seen[p as usize] || faulty(Fault::DrawDuplicates) {
                seen[p as usize] = true;
                out.push(p);
            }
        }
    }
    Some((out, stream))
}

/// Per-generator positions to per-shot generator lists, ascending.
fn transpose(
    pos: &[u32],
    stride: usize,
    counts: &[u32],
    n_shots: usize,
) -> (Vec<u32>, Vec<u32>, u64) {
    let mut per_shot = vec![0u32; n_shots];
    for (g, &count) in counts.iter().enumerate() {
        for &p in &pos[g * stride..g * stride + count as usize] {
            per_shot[p as usize] += 1;
        }
    }
    let mut offsets = vec![0u32; n_shots + 1];
    for s in 0..n_shots {
        offsets[s + 1] = offsets[s] + per_shot[s];
    }
    let mut cursor = offsets.clone();
    let mut indices = vec![0u32; offsets[n_shots] as usize];
    for (g, &count) in counts.iter().enumerate() {
        for &p in &pos[g * stride..g * stride + count as usize] {
            indices[cursor[p as usize] as usize] = g as u32;
            cursor[p as usize] += 1;
        }
    }
    let nonempty = per_shot.iter().filter(|&&c| c != 0).count() as u64;
    (offsets, indices, nonempty)
}

/// Accepted shots and, of those, logical errors. Shots where nothing fired
/// are left to the caller, and shots whose generators cancel are dropped.
fn score(
    rows: &[u64],
    stride: usize,
    planes_end: usize,
    post_end: usize,
    offsets: &[u32],
    indices: &[u32],
) -> (u64, u64) {
    let mut accepted = 0;
    let mut logical = 0;
    let mut acc = vec![0u64; stride];
    for shot in offsets.windows(2) {
        if shot[0] == shot[1] {
            continue;
        }
        acc.iter_mut().for_each(|w| *w = 0);
        for &g in &indices[shot[0] as usize..shot[1] as usize] {
            let row = &rows[g as usize * stride..(g as usize + 1) * stride];
            acc.iter_mut().zip(row).for_each(|(a, r)| *a ^= r);
        }
        let identity = acc[..planes_end].iter().all(|&w| w == 0);
        let detected = acc[planes_end..post_end].iter().any(|&w| w != 0);
        if !identity && !detected {
            accepted += 1;
            if acc[post_end..].iter().any(|&w| w != 0) {
                logical += 1;
            }
        }
    }
    if accepted > 0 && faulty(Fault::LogicalCount) {
        logical += 1;
    }
    (accepted, logical)
}

unsafe extern "C" fn ler_score(ctx: *mut Ctx, batch: *const LerBatch) -> c_int {
    if ctx.is_null() || batch.is_null() {
        return -1;
    }
    let b = unsafe { &*batch };
    if b.n_shots <= 0
        || b.n_generators <= 0
        || b.n_indices < 0
        || b.stride <= 0
        || b.stride > LER_MAX_WORDS
    {
        return -1;
    }
    let stride = b.stride as usize;
    let rows = unsafe { slice(b.gen_words, b.n_generators as usize * stride) };
    let offsets = unsafe { slice(b.offsets, b.n_shots as usize + 1) };
    let indices = unsafe { slice(b.indices, b.n_indices as usize) };
    let (accepted, logical) = score(
        rows,
        stride,
        b.planes_end as usize,
        b.post_end as usize,
        offsets,
        indices,
    );
    unsafe {
        *b.out_accepted = accepted;
        *b.out_logical = logical;
    }
    0
}

unsafe extern "C" fn ler_draw_parity(ctx: *mut Ctx, req: *const LerDrawParity) -> c_int {
    if ctx.is_null() || req.is_null() {
        return -1;
    }
    let r = unsafe { &*req };
    if r.n_generators == 0 || r.n_tables == 0 || r.n_cdf == 0 || r.pos_stride == 0 || r.nshots == 0
    {
        return -1;
    }
    let n_gen = r.n_generators as usize;
    let tables = unsafe {
        Tables {
            cdf: slice(r.cdf_all, r.n_cdf as usize),
            offset: slice(r.table_offset, r.n_tables as usize),
            len: slice(r.table_len, r.n_tables as usize),
            klo: slice(r.table_klo, r.n_tables as usize),
            of: slice(r.table_of, n_gen),
        }
    };
    let probe = r.probe_words as usize;
    let stride = r.pos_stride as usize;
    let words = unsafe { slice_mut(r.out_words, probe * n_gen) };
    let counts = unsafe { slice_mut(r.out_counts, n_gen) };
    let pos = unsafe { slice_mut(r.out_pos, stride * n_gen) };
    let after = unsafe { slice_mut(r.out_after, n_gen) };
    for g in 0..n_gen {
        let mut fresh = GeneratorStream::new(r.seed, g);
        for w in &mut words[g * probe..(g + 1) * probe] {
            *w = fresh.next_u64();
        }
        match draw(r.seed, g, &tables, r.nshots, r.pos_stride) {
            Some((drawn, mut stream)) => {
                counts[g] = drawn.len() as u32;
                pos[g * stride..g * stride + drawn.len()].copy_from_slice(&drawn);
                after[g] = stream.next_u64();
            }
            None => counts[g] = LER_DRAW_OVERFLOW,
        }
    }
    0
}

unsafe extern "C" fn ler_transpose(ctx: *mut Ctx, req: *const LerTranspose) -> c_int {
    if ctx.is_null() || req.is_null() {
        return -1;
    }
    let r = unsafe { &*req };
    if r.n_generators == 0
        || r.n_generators > LER_MAX_GENERATORS
        || r.n_shots == 0
        || r.pos_stride == 0
    {
        return -1;
    }
    let n_gen = r.n_generators as usize;
    let stride = r.pos_stride as usize;
    let counts = unsafe { slice(r.gen_counts, n_gen) };
    if counts.iter().any(|&c| c > r.pos_stride) {
        return -1;
    }
    // The transpose overwrites the buffers a resident sample would live in.
    unsafe { session(ctx) }.resident = None;
    let pos = unsafe { slice(r.pos, stride * n_gen) };
    let (offsets, indices, nonempty) = transpose(pos, stride, counts, r.n_shots as usize);
    unsafe { slice_mut(r.out_offsets, offsets.len()) }.copy_from_slice(&offsets);
    if indices.len() > r.n_indices_capacity as usize {
        return -1;
    }
    unsafe { slice_mut(r.out_indices, indices.len()) }.copy_from_slice(&indices);
    unsafe { *r.out_nonempty = nonempty };
    0
}

unsafe extern "C" fn ler_sample_score(ctx: *mut Ctx, req: *const LerSampleScore) -> c_int {
    if ctx.is_null() || req.is_null() {
        return -1;
    }
    let r = unsafe { &*req };
    if r.n_generators <= 0
        || r.n_generators as u32 > LER_MAX_GENERATORS
        || r.n_shots <= 0
        || r.stride <= 0
        || r.stride > LER_MAX_WORDS
        || r.planes_end < 0
        || r.post_end < r.planes_end
        || r.post_end > r.stride
        || r.pos_stride == 0
    {
        return -1;
    }
    let session = unsafe { session(ctx) };
    let n_gen = r.n_generators as usize;
    let n_shots = r.n_shots as u32;
    let reuse = r.reuse_sample != 0;
    if reuse {
        match &session.resident {
            Some(s) if s.shots == n_shots && s.generators == n_gen as u32 => {}
            _ => return -1,
        }
    } else {
        if r.n_tables == 0 || r.n_cdf == 0 {
            return -1;
        }
        session.resident = None;
        let tables = unsafe {
            Tables {
                cdf: slice(r.cdf_all, r.n_cdf as usize),
                offset: slice(r.table_offset, r.n_tables as usize),
                len: slice(r.table_len, r.n_tables as usize),
                klo: slice(r.table_klo, r.n_tables as usize),
                of: slice(r.table_of, n_gen),
            }
        };
        let stride = r.pos_stride as usize;
        let mut pos = vec![0u32; stride * n_gen];
        let mut counts = vec![0u32; n_gen];
        for g in 0..n_gen {
            match draw(r.seed, g, &tables, n_shots, r.pos_stride) {
                Some((drawn, _)) => {
                    counts[g] = drawn.len() as u32;
                    pos[g * stride..g * stride + drawn.len()].copy_from_slice(&drawn);
                }
                // A draw that does not fit fails the whole dispatch.
                None => return -1,
            }
        }
        let (offsets, indices, nonempty) = transpose(&pos, stride, &counts, n_shots as usize);
        unsafe { *r.out_nonempty = nonempty };
        session.resident = Some(Resident {
            shots: n_shots,
            generators: n_gen as u32,
            offsets,
            indices,
        });
    }
    let sample = session.resident.as_ref().expect("a sample is resident");
    let stride = r.stride as usize;
    let rows = unsafe { slice(r.gen_words, n_gen * stride) };
    let (accepted, logical) = score(
        rows,
        stride,
        r.planes_end as usize,
        r.post_end as usize,
        &sample.offsets,
        &sample.indices,
    );
    unsafe {
        *r.out_accepted = accepted;
        *r.out_logical = logical;
    }
    0
}

/// Registers the reference engine. Once per process.
pub fn register() {
    unsafe { _internal_r::engine::register_engine((&REFERENCE as *const Engine).cast()) }
        .expect("the engine registers");
}

/// Runs every conformance check, returning `(check, outcome, detail)` with
/// `outcome` one of passed, failed, skipped.
pub fn run() -> Vec<(&'static str, &'static str, String)> {
    _internal_r::engine::run_conformance()
        .into_iter()
        .map(|(name, outcome)| match outcome {
            _internal_r::engine::CheckOutcome::Passed => (name, "passed", String::new()),
            _internal_r::engine::CheckOutcome::Failed(d) => (name, "failed", d),
            _internal_r::engine::CheckOutcome::Skipped(d) => (name, "skipped", d),
        })
        .collect()
}
