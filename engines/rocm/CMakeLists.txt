cmake_minimum_required(VERSION 3.21)

# GPU_TARGETS (default gfx950) takes a semicolon- or comma-separated list of
# offload targets for a fat binary.
if(GPU_TARGETS)
  string(REPLACE "," ";" GPU_TARGETS "${GPU_TARGETS}")
  set(CMAKE_HIP_ARCHITECTURES "${GPU_TARGETS}")
endif()

if(DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH "$ENV{ROCM_PATH}")
else()
  set(ROCM_PATH /opt/rocm)
endif()
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")

project(qiskit_paulice_rocm LANGUAGES C CXX HIP)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.SABIModule)
find_package(hip REQUIRED)
find_package(rocprim REQUIRED)

# The ABI header is qiskit-paulice's own, two levels up, so the engine builds
# from a checkout of this repository and not from an sdist.
set(PAULICE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
if(NOT EXISTS "${PAULICE_INCLUDE_DIR}/paulice_engine.h")
  message(FATAL_ERROR
    "paulice_engine.h not found in ${PAULICE_INCLUDE_DIR}; build from a qiskit-paulice checkout")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)

set(HIP_SOURCES
  src/paulice_rocm.cpp
  src/coverage.hip.cpp
  src/ler.hip.cpp
  src/draw.hip.cpp
  src/transpose.hip.cpp)
set_source_files_properties(${HIP_SOURCES} PROPERTIES LANGUAGE HIP)

Python_add_library(_engine MODULE USE_SABI 3.10 WITH_SOABI
  src/module.c src/engine_table.cpp ${HIP_SOURCES})
target_include_directories(_engine PRIVATE "${PAULICE_INCLUDE_DIR}" src)
target_link_libraries(_engine PRIVATE hip::host roc::rocprim)
# The engine reproduces the host's floating-point accumulation exactly, which
# reassociation would break, and so would fused multiply-adds, which round
# once where the host rounds twice.
target_compile_options(_engine PRIVATE
  $<$<COMPILE_LANGUAGE:HIP>:-O3 -ffp-contract=off>
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -ffp-contract=off>)
set_target_properties(_engine PROPERTIES
  INSTALL_RPATH "${ROCM_PATH}/lib")

install(TARGETS _engine LIBRARY DESTINATION qiskit_paulice_rocm)
