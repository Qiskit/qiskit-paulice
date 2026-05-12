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

//! Vendored from `rustiq_core::routines::decoding` (v0.0.9), with one change:
//! `shuffle_parities` (and therefore `information_set_decoding`) now takes a
//! caller-supplied `RngCore` instead of using `rand::thread_rng()`. This is what
//! lets us drive the entire check-picking pipeline reproducibly from a single
//! seed -- the upstream `thread_rng()` is OS-seeded and not externally
//! re-seedable, which made cross-process reproducibility impossible.

use rand::RngCore;
use rand::seq::SliceRandom;

pub fn syndrome_decoding(parities: &[Vec<bool>], input_target: &Vec<bool>) -> Option<Vec<bool>> {
    let mut target = input_target.clone();
    let mut solution = vec![false; parities.len()];
    let mut hweight = target.iter().filter(|a| **a).count();
    loop {
        let mut best_reduction = 0;
        let mut best_index: i32 = -1;
        for (index, parity) in parities.iter().enumerate() {
            let new_hweight = parity
                .iter()
                .zip(target.iter())
                .map(|(a, b)| a ^ b)
                .filter(|a| *a)
                .count();
            if (hweight as i32 - new_hweight as i32) > best_reduction {
                best_reduction = hweight as i32 - new_hweight as i32;
                best_index = index as i32;
            }
        }
        if best_index == -1 {
            break;
        }
        for (a, b) in target.iter_mut().zip(parities[best_index as usize].iter()) {
            *a ^= b;
        }
        solution[best_index as usize] ^= true;
        hweight = target.iter().filter(|a| **a).count();
    }
    let mut true_target = vec![false; target.len()];
    for (i, b) in solution.iter().enumerate() {
        if *b {
            for (x, y) in parities[i].iter().zip(true_target.iter_mut()) {
                *y ^= x;
            }
        }
    }
    if true_target != *input_target {
        return None;
    }
    Some(solution)
}

fn colop(parities: &mut [Vec<bool>], i: usize, j: usize) {
    for row in parities.iter_mut() {
        row[j] ^= row[i];
    }
}

fn shuffle_parities<R: RngCore + ?Sized>(
    parities: &mut Vec<Vec<bool>>,
    target: &mut [bool],
    row_ech: bool,
    rng: &mut R,
) -> Vec<usize> {
    let n = parities.first().unwrap().len();
    let mut row_permutation: Vec<usize> = (0..parities.len()).collect();
    row_permutation.shuffle(rng);
    let mut new_parities = Vec::new();
    for j in row_permutation.iter() {
        new_parities.push(parities[*j].clone());
    }
    if row_ech {
        let mut rank = 0;
        for i in 0..parities.len() {
            let mut pivot = None;
            for j in rank..n {
                if new_parities[i][j] {
                    pivot = Some(j);
                    break;
                }
            }
            if let Some(pivot) = pivot {
                if pivot != rank {
                    colop(&mut new_parities, pivot, rank);
                    target[rank] ^= target[pivot];
                }
                for j in 0..n {
                    if new_parities[i][j] && j != rank {
                        colop(&mut new_parities, rank, j);
                        target[j] ^= target[rank];
                    }
                }
                rank += 1;
                if rank == n {
                    break;
                }
            }
        }
    }
    *parities = new_parities;
    row_permutation
}

fn fix_permutation(solution: &[bool], permutation: &[usize]) -> Vec<bool> {
    let mut new_solution = vec![false; solution.len()];
    for (i, j) in permutation.iter().enumerate() {
        new_solution[*j] = solution[i];
    }
    new_solution
}

pub fn information_set_decoding<R: RngCore + ?Sized>(
    input_parities: &[Vec<bool>],
    input_target: &Vec<bool>,
    ntries: usize,
    row_ech: bool,
    rng: &mut R,
) -> Option<Vec<bool>> {
    let mut best_solution = None;
    let mut best_cost = None;
    for _ in 0..ntries {
        let mut parities = input_parities.to_owned();
        let mut target = input_target.clone();
        let permutation = shuffle_parities(&mut parities, &mut target, row_ech, rng);
        let solution = syndrome_decoding(&parities, &target);
        if let Some(solution) = solution {
            let solution = fix_permutation(&solution, &permutation);
            let cost = solution.iter().filter(|a| **a).count();
            if let Some(best_cost) = best_cost {
                if best_cost < cost {
                    continue;
                }
            }
            best_cost = Some(cost);
            best_solution = Some(solution);
        }
    }
    if let Some(solution) = best_solution {
        let mut true_target = vec![false; input_target.len()];
        for (i, b) in solution.iter().enumerate() {
            if *b {
                for (x, y) in input_parities[i].iter().zip(true_target.iter_mut()) {
                    *y ^= x;
                }
            }
        }
        assert_eq!(true_target, *input_target);
        return Some(solution);
    }
    None
}
