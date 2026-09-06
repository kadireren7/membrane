cmake_minimum_required(VERSION 3.16)
project(membrane C)

# Phase 17: standard, distro-correct install destinations (bin/, lib/ or
# lib64/, share/, ...) -- used by tools/membrane-run/CMakeLists.txt's
# install() rules instead of hardcoded "bin"/"lib" strings.
include(GNUInstallDirs)

# Phase 17: CMake has no built-in "uninstall" target. This one removes
# exactly the files install_manifest.txt recorded for *this* build
# directory's own install run -- never a directory, never a guess.
# Registered unconditionally (not only under MEMBRANE_ENABLE_LLAMA) so
# `cmake --build <dir> --target uninstall` gives the same clear,
# manifest-driven error either way if nothing was installed yet.
if(NOT TARGET uninstall)
  configure_file(
    ${CMAKE_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in
    ${CMAKE_BINARY_DIR}/cmake_uninstall.cmake
    IMMEDIATE @ONLY)
  add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_BINARY_DIR}/cmake_uninstall.cmake)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(MEMBRANE_ENABLE_SANITIZERS
    "Enable AddressSanitizer and UndefinedBehaviorSanitizer"
    OFF)
option(MEMBRANE_ENABLE_TSAN
    "Enable ThreadSanitizer (mutually exclusive with the ASan build)"
    OFF)
option(MEMBRANE_ENABLE_ASAN
    "Deprecated alias for MEMBRANE_ENABLE_SANITIZERS"
    OFF)
option(MEMBRANE_BUILD_TESTS "Build unit tests" ON)
option(MEMBRANE_ENABLE_LLAMA
    "Build the llama.cpp KV capture adapter (needs the submodule)"
    OFF)

if(MEMBRANE_ENABLE_ASAN)
  message(DEPRECATION
    "MEMBRANE_ENABLE_ASAN is deprecated; use MEMBRANE_ENABLE_SANITIZERS")
  set(MEMBRANE_ENABLE_SANITIZERS ON)
endif()

if(MEMBRANE_ENABLE_SANITIZERS AND MEMBRANE_ENABLE_TSAN)
  message(FATAL_ERROR
    "MEMBRANE_ENABLE_TSAN cannot be combined with the ASan/UBSan build")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

# Sanitizer flags travel as an interface target so every executable that
# links membrane_core inherits them for both compile and link steps.
add_library(membrane_sanitizers INTERFACE)
if(MEMBRANE_ENABLE_SANITIZERS)
  if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(membrane_sanitizers INTERFACE
      -fsanitize=address,undefined
      -fno-omit-frame-pointer
      -g)
    target_link_options(membrane_sanitizers INTERFACE
      -fsanitize=address,undefined)
  else()
    message(WARNING
      "MEMBRANE_ENABLE_SANITIZERS is ON but compiler "
      "'${CMAKE_C_COMPILER_ID}' is not GCC/Clang; flags not applied")
  endif()
endif()

if(MEMBRANE_ENABLE_TSAN)
  if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(membrane_sanitizers INTERFACE
      -fsanitize=thread
      -fno-omit-frame-pointer
      -g)
    target_link_options(membrane_sanitizers INTERFACE
      -fsanitize=thread)
  else()
    message(WARNING
      "MEMBRANE_ENABLE_TSAN is ON but compiler "
      "'${CMAKE_C_COMPILER_ID}' is not GCC/Clang; flags not applied")
  endif()
endif()

find_package(Threads REQUIRED)
# Phase 6.4: zlib for attntrace2.c's optional DEFLATE compression --
# ubiquitous system library (comparable to libm in availability),
# added specifically to fix Phase 6.3's real committed-trace-file-size
# problem (docs/phase6-exact-sparse-retrieval.md section 2/17) at the
# root rather than trading away capture resolution again.
find_package(ZLIB REQUIRED)

# Phase 17: explicit STATIC, not BUILD_SHARED_LIBS-controlled like
# llama/ggml further down. Investigated directly rather than left
# implicit: this add_library() call runs before third_party/llama.cpp's
# own CMake ever sets BUILD_SHARED_LIBS=ON (that only happens inside
# the MEMBRANE_ENABLE_LLAMA block, much later in this file), so on the
# exact documented fresh-build-directory workflow membrane_core was
# ALREADY deterministically static -- confirmed via the real generated
# link command (ar, not cc -shared) on a from-scratch configure. But an
# explicit -DBUILD_SHARED_LIBS=ON on the command line, or reconfiguring
# an existing build directory whose cache already has it ON from some
# earlier configure, bypasses that ordering and makes it genuinely
# shared instead -- also confirmed directly. No dlopen/dlsym or export
# config anywhere in this repository depends on membrane_core being
# loadable as a shared object, and the product doesn't expose it as a
# development SDK, so there's no reason for its type to depend on
# incidental configure-order/cache-history at all. Pinning it removes
# that footgun instead of merely tolerating it (tools/membrane-run/
# CMakeLists.txt's install loop's TYPE check still handles a
# hypothetical future SHARED membrane_core correctly, but should now
# never actually see one for this target).
add_library(membrane_core STATIC
  src/block/block.c
  src/codecs/raw.c
  src/codecs/rle.c
  src/codecs/f16_byteplane.c
  src/codecs/f16_xor_byteplane.c
  src/codecs/huffman.c
  src/codecs/f16_highplane_huffman.c
  src/codecs/f16convert.c
  src/codecs/q8block.c
  src/codecs/codec_registry.c
  src/stats.c
  src/store/store.c
  src/store/store_index.c
  src/backends/backend.c
  src/backends/backend_file.c
  src/kvdump/kvdump.c
  src/kvdump/kvmetrics.c
  src/kvdump/kvpredict.c
  src/kvdump/kvquant.c
  src/hash/sha256.c
  src/policy/policy.c
  src/quant/quant_simd.c
  src/quant/quant_select.c
  src/fpga/fpga_dma.c
  src/fpga/quant_backend.c
  src/kvtrace/kvtrace.c
  src/attntrace/attntrace.c
  src/attntrace/attntrace2.c
  src/attntrace/attntrace3.c
)
target_include_directories(membrane_core PUBLIC include)
target_link_libraries(membrane_core PUBLIC membrane_sanitizers Threads::Threads m ZLIB::ZLIB)

add_subdirectory(tools/membrane-bench)
add_subdirectory(tools/membrane-store-bench)
add_subdirectory(tools/membrane-kv-analyze)
add_subdirectory(tools/membrane-quant-bench)
add_subdirectory(tools/membrane-hw-sim)
add_subdirectory(tools/membrane-workload-core)
add_subdirectory(tools/membrane-demo)
add_subdirectory(tools/membrane-quant-policy-bench)
add_subdirectory(tools/membrane-kv-trace-export)

# Product Phase 5: the llama-free core of the live shadow runtime
# (block packing/tail accounting, telemetry aggregation, token-sequence
# comparison -- see tools/membrane-llama-runtime/runtime_core.h). Built
# unconditionally, like membrane_bench_core/membrane_export_core above,
# so it and its tests run under the default llama-free build; only the
# llama-facing executable (tools/membrane-llama-runtime/CMakeLists.txt,
# main.cpp + llama_hook.cpp) needs MEMBRANE_ENABLE_LLAMA=ON, added
# further down alongside the other llama-gated tool subdirectories.
add_library(membrane_runtime_core STATIC
  tools/membrane-llama-runtime/runtime_core.c)
target_include_directories(membrane_runtime_core PUBLIC
  tools/membrane-llama-runtime)
target_link_libraries(membrane_runtime_core PUBLIC membrane_core
  membrane_workload_core)

# Product Phase 7: llama-free core for compressed-KV-storage measurement
# (RSS checkpoints, byte accounting, telemetry) -- see the file header
# in kv_store_telemetry.h for why this stays free of ggml/llama types,
# same rationale as membrane_runtime_core above. Built unconditionally
# so test_kv_store_telemetry runs under the default llama-free build.
add_library(membrane_kv_store_telemetry STATIC
  tools/membrane-llama-runtime/kv_store_telemetry.c)
target_include_directories(membrane_kv_store_telemetry PUBLIC
  tools/membrane-llama-runtime)

# membrane-llama-run's CLI argument parser (tools/membrane-llama-runtime/
# cli_parse.h/.cpp), split out of main.cpp so it -- like
# membrane_runtime_core above -- builds and is testable (test_cli_parse)
# under the default llama-free configuration, with no model load and no
# llama.cpp dependency anywhere in this library.
enable_language(CXX)
add_library(membrane_cli_parse STATIC
  tools/membrane-llama-runtime/cli_parse.cpp)
set_target_properties(membrane_cli_parse PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)
target_link_libraries(membrane_cli_parse PUBLIC membrane_runtime_core
  membrane_kv_store_telemetry)

# Product Phase 8: membrane-run's product CLI parser and compatibility
# pre-check -- both llama-free (no model load, no ggml/llama header),
# built unconditionally like the libraries above.
add_library(membrane_product_cli STATIC
  tools/membrane-run/product_cli.cpp)
set_target_properties(membrane_product_cli PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)
target_link_libraries(membrane_product_cli PUBLIC membrane_cli_parse)
target_include_directories(membrane_product_cli PUBLIC tools/membrane-run)

add_library(membrane_compat_check STATIC
  tools/membrane-run/compat_check.c)
target_include_directories(membrane_compat_check PUBLIC tools/membrane-run)

# Phase 9B.1: --gpu-layers auto's pure arithmetic, deliberately
# llama-free (no ggml/llama header, no device access) so it's testable
# without a GPU -- same pattern as membrane_compat_check above.
add_library(membrane_gpu_policy STATIC
  tools/membrane-run/gpu_policy.c)
target_include_directories(membrane_gpu_policy PUBLIC tools/membrane-run)

# Phase 11A: --kv adaptive's pure Q8-vs-Q5 decision, deliberately
# llama-free and independent of gpu_policy's own arithmetic (it only
# consumes gpu_policy_resolve()'s already-computed results, passed in
# by the caller) -- same testable-without-a-model/GPU pattern as
# membrane_compat_check/membrane_gpu_policy above.
add_library(membrane_adaptive_kv_policy STATIC
  tools/membrane-run/adaptive_kv_policy.c)
target_include_directories(membrane_adaptive_kv_policy PUBLIC
  tools/membrane-run)

# Phase 12H: --kv-placement's pure static KV residency planner --
# llama-free, no device/model access, same testable-without-a-GPU
# pattern as membrane_gpu_policy/membrane_adaptive_kv_policy above.
# Independent of gpu_policy.c's own arithmetic (never recomputes or
# feeds back into weight-layer selection -- see kv_residency_policy.h).
add_library(membrane_kv_residency_policy STATIC
  tools/membrane-run/kv_residency_policy.c)
target_include_directories(membrane_kv_residency_policy PUBLIC
  tools/membrane-run)

# Phase 20: the joint GPU-layers x KV-precision planner -- llama-free,
# same testable-without-a-GPU pattern as the four modules above, whose
# pure logic it calls (never duplicates): compat_check for
# architecture gating, gpu_policy for the safety-reserve formula,
# adaptive_kv_policy for the Q8-vs-Q5 decision, kv_residency_policy for
# placement feasibility.
add_library(membrane_joint_planner STATIC
  tools/membrane-run/joint_planner.c)
# CodeRabbit review (PR #31): membrane_sanitizers is the INTERFACE
# library that actually carries the -fsanitize=... compile/link flags
# (see its own definition above) -- without linking it here, a
# MEMBRANE_ENABLE_SANITIZERS/MEMBRANE_ENABLE_TSAN configure builds this
# target with no instrumentation at all, silently. PUBLIC so
# membrane_auto_fallback (which itself only needs this header's plain-
# data type, not this library's own object code) still gets
# instrumented transitively wherever it links membrane_joint_planner in
# a test binary.
target_link_libraries(membrane_joint_planner PUBLIC membrane_compat_check
  membrane_gpu_policy membrane_adaptive_kv_policy
  membrane_kv_residency_policy membrane_sanitizers)
target_include_directories(membrane_joint_planner PUBLIC tools/membrane-run)

# Phase 34: real host-RAM capacity check -- llama-free, no /proc read
# (Section 9 of the Phase 34 task: reuse main.cpp's single existing
# read_host_meminfo(), never a second path), same testable-without-a-
# model/GPU pattern as gpu_policy.h.
add_library(membrane_host_memory_guard STATIC
  tools/membrane-run/host_memory_guard.c)
target_link_libraries(membrane_host_memory_guard PUBLIC membrane_sanitizers)
target_include_directories(membrane_host_memory_guard PUBLIC
  tools/membrane-run)

# Phase 33: safe context recommendation's pure/testable core -- llama-
# free (no ggml/llama header, no model load, no device access, no
# /proc read), same testable-without-a-model/GPU pattern as
# membrane_joint_planner above, whose exact, unchanged
# membrane_joint_plan_resolve() it calls once per bounded candidate
# context size (never a second planner -- see context_recommender.h's
# own top comment). No CLI surface yet (--ctx auto ships in Phase 34).
# Phase 34: also links membrane_host_memory_guard -- every candidate
# the joint planner accepts is then checked against real host RAM
# availability before being counted feasible.
add_library(membrane_context_recommender STATIC
  tools/membrane-run/context_recommender.c)
target_link_libraries(membrane_context_recommender PUBLIC
  membrane_joint_planner membrane_host_memory_guard membrane_sanitizers)
target_include_directories(membrane_context_recommender PUBLIC
  tools/membrane-run)

# Phase 35: --ctx auto's own pure/testable CLI-adapter pieces --
# minimum-required-context arithmetic and error-suggestion mapping.
# llama-free, no model/device/host access (Section 14 of the Phase 35
# task: "Do not put planning logic into CLI parsing" -- the llama-aware
# fact-gathering itself lives in main.cpp, which calls this module with
# already-gathered plain data).
add_library(membrane_context_auto_cli STATIC
  tools/membrane-run/context_auto_cli.c)
target_link_libraries(membrane_context_auto_cli PUBLIC
  membrane_context_recommender membrane_host_memory_guard
  membrane_sanitizers)
target_include_directories(membrane_context_auto_cli PUBLIC
  tools/membrane-run)

# Phase 21: the bounded apply-time fallback CONTROLLER -- llama-free,
# same pattern as membrane_joint_planner above. Consumes (never
# duplicates or regenerates) the ranked candidate array
# membrane_joint_plan_resolve() already produced; only depends on
# joint_planner.h's plain-data candidate type, not the planner library
# itself (no candidate GENERATION happens here).
add_library(membrane_auto_fallback STATIC
  tools/membrane-run/auto_fallback.c)
target_link_libraries(membrane_auto_fallback PUBLIC membrane_sanitizers)
target_include_directories(membrane_auto_fallback PUBLIC tools/membrane-run)

# membrane-cxl-sim (Phase 6.1) is a pure discrete-event simulator, no
# llama.cpp dependency -- built unconditionally like the C tools above,
# unlike tools/membrane-kv-capture etc, which need MEMBRANE_ENABLE_LLAMA.
enable_language(CXX)
add_subdirectory(tools/membrane-cxl-sim)

# membrane-kv-workingset-sim (Phase 6.2) is also a pure discrete-event
# simulator with no llama.cpp dependency -- built unconditionally like
# membrane-cxl-sim, whose sim_config.h calibration constants it reuses
# via an include path only (no link dependency on the cxl-sim binary).
add_subdirectory(tools/membrane-kv-workingset-sim)

# membrane-kv-exact-sim (Phase 6.3) merges membrane-cxl-sim's shared
# k_server_resource_t contention model with membrane-kv-workingset-sim's
# per-channel predictor/hot-cache engine -- also no llama.cpp
# dependency, built unconditionally.
add_subdirectory(tools/membrane-kv-exact-sim)

if(MEMBRANE_ENABLE_LLAMA)
  enable_language(CXX)
  if(NOT EXISTS ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/CMakeLists.txt)
    message(FATAL_ERROR "llama.cpp submodule missing; run: "
      "git submodule update --init third_party/llama.cpp")
  endif()

  # The MEMBRANE per-layer KV type override needs a small patch to the
  # pinned llama.cpp commit (see
  # patches/llama.cpp-membrane-kv-type-override.patch and
  # docs/phase4-runtime-policy.md). Applied here, idempotently, so a
  # fresh submodule checkout builds without a manual step -- the
  # submodule's own commit pin is never changed, only its working tree.
  #
  # Idempotency check (Section 15): a content-marker grep for
  # "kv_type_override" in include/llama.h, NOT `git apply --reverse
  # --check` -- Phase 12H discovered directly that reverse-checking an
  # EARLIER patch in a stack spuriously fails once a LATER patch (kv-
  # device-override, applied right below) has also been applied: the
  # later patch's own context lines lean on the earlier patch's added
  # lines, and `git apply --reverse --check` isn't reliable across that
  # interleaving. A simple "does the file already contain what this
  # patch adds" grep has no such failure mode and is deterministic.
  set(MEMBRANE_LLAMA_PATCH
    ${CMAKE_SOURCE_DIR}/patches/llama.cpp-membrane-kv-type-override.patch)
  execute_process(
    COMMAND grep -q "kv_type_override"
      ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/include/llama.h
    RESULT_VARIABLE MEMBRANE_PATCH_ALREADY_APPLIED)
  if(NOT MEMBRANE_PATCH_ALREADY_APPLIED EQUAL 0)
    execute_process(
      COMMAND git apply ${MEMBRANE_LLAMA_PATCH}
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/llama.cpp
      RESULT_VARIABLE MEMBRANE_PATCH_APPLY_RESULT)
    if(NOT MEMBRANE_PATCH_APPLY_RESULT EQUAL 0)
      message(FATAL_ERROR "failed to apply ${MEMBRANE_LLAMA_PATCH} to "
        "third_party/llama.cpp -- see docs/phase4-runtime-policy.md")
    endif()
    message(STATUS "applied MEMBRANE kv_type_override patch to llama.cpp")
  endif()

  # Phase 12H: --kv-placement's static per-layer KV DEVICE override
  # needs a second small patch on top of kv_type_override above (see
  # patches/llama.cpp-membrane-kv-device-override.patch and
  # docs/kv-residency.md). Deliberately the SAME two-patch set proven
  # safe in Phase 12B-12G research -- the kv-runtime-relocate and
  # kv-buffer-retirement patches are NOT applied here: static placement
  # is decided once at context-construction time and never revisited
  # during decode, so this product path has no use for runtime
  # relocation or its buffer-retirement cleanup (both stay research-
  # only, on their own experiment branches -- Section 14/44: minimal
  # product patch surface is a hard requirement, not a convenience).
  # Same content-marker idempotency check as kv_type_override above.
  set(MEMBRANE_LLAMA_PATCH_DEVICE_OVERRIDE
    ${CMAKE_SOURCE_DIR}/patches/llama.cpp-membrane-kv-device-override.patch)
  execute_process(
    COMMAND grep -q "kv_dev_override"
      ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/include/llama.h
    RESULT_VARIABLE MEMBRANE_DEVICE_PATCH_ALREADY_APPLIED)
  if(NOT MEMBRANE_DEVICE_PATCH_ALREADY_APPLIED EQUAL 0)
    execute_process(
      COMMAND git apply ${MEMBRANE_LLAMA_PATCH_DEVICE_OVERRIDE}
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/llama.cpp
      RESULT_VARIABLE MEMBRANE_DEVICE_PATCH_APPLY_RESULT)
    if(NOT MEMBRANE_DEVICE_PATCH_APPLY_RESULT EQUAL 0)
      message(FATAL_ERROR "failed to apply "
        "${MEMBRANE_LLAMA_PATCH_DEVICE_OVERRIDE} to third_party/llama.cpp "
        "-- see docs/kv-residency.md")
    endif()
    message(STATUS "applied MEMBRANE kv_dev_override patch to llama.cpp")
  endif()

  # Phase 17: fail clearly, at configure time, before descending into
  # third_party/llama.cpp's own CMake -- which itself does
  # find_package(Vulkan COMPONENTS glslc REQUIRED) deep inside
  # ggml/src/ggml-vulkan/CMakeLists.txt and would otherwise be the first
  # place a missing Vulkan SDK surfaces, with a message that never
  # mentions MEMBRANE or points back to the README's documented
  # packages. GGML_VULKAN is llama.cpp's own option name (not defined
  # by this file), read here only to decide whether to run this
  # MEMBRANE-level pre-check -- it does not change llama.cpp's own
  # handling of the flag.
  if(GGML_VULKAN)
    find_package(Vulkan QUIET)
    # Vulkan_glslc_FOUND/Vulkan::glslc need CMake 3.24/3.19 respectively
    # (this project requires only 3.16 -- see cmake_minimum_required
    # above), so find_package(Vulkan) alone can't tell us whether glslc
    # is actually present; a Vulkan SDK install missing only the shader
    # compiler would otherwise pass this pre-check and still fail
    # inside third_party/llama.cpp's own find_package(Vulkan COMPONENTS
    # glslc REQUIRED) -- exactly the generic, MEMBRANE-unaware error
    # this whole pre-check exists to avoid. find_program() works the
    # same on every supported CMake version.
    find_program(MEMBRANE_GLSLC_EXECUTABLE NAMES glslc
      HINTS "$ENV{VULKAN_SDK}/bin" "$ENV{VULKAN_SDK}/Bin")
    if(NOT Vulkan_FOUND OR NOT MEMBRANE_GLSLC_EXECUTABLE)
      message(FATAL_ERROR
        "Vulkan backend requested (-DGGML_VULKAN=ON) but Vulkan "
        "development files were not found. Install the Vulkan SDK "
        "development headers, the glslc shader compiler, and SPIR-V "
        "headers for your distribution (see README.md's Vulkan build "
        "section for the packages confirmed necessary on Ubuntu/"
        "Pop!_OS), then re-run cmake.")
    endif()
  endif()

  # Mega Phase D, PR D3: same real-toolchain, MEMBRANE-specific-error
  # pre-check as the Vulkan block above, for the same reason -- fail
  # here, before descending into third_party/llama.cpp's own ggml-cuda/
  # CMakeLists.txt (which does its own find_package(CUDAToolkit), with
  # a message that never mentions MEMBRANE). GGML_CUDA is llama.cpp's
  # own option name (not defined by this file), read here only to
  # decide whether to run this pre-check -- it does not change
  # llama.cpp's own handling of the flag.
  if(GGML_CUDA)
    find_package(CUDAToolkit QUIET)
    if(NOT CUDAToolkit_FOUND)
      message(FATAL_ERROR
        "CUDA backend requested (-DGGML_CUDA=ON) but no CUDA Toolkit "
        "was found. Install the NVIDIA CUDA Toolkit (nvcc + cudart + "
        "cublas) for your distribution, or point CMake at a non-"
        "standard install with -DCUDAToolkit_ROOT=/path/to/toolkit "
        "(see docs/cuda-backend.md), then re-run cmake. A real NVIDIA "
        "GPU + driver alone (nvidia-smi working) is NOT sufficient --"
        "the compiler toolkit is a separate install.")
    endif()
  endif()

  # Mega Phase D, PR D4: same real-toolchain, MEMBRANE-specific-error
  # pre-check pattern as the Vulkan/CUDA blocks above -- but Metal has
  # no "development files missing" middle ground the way Vulkan/CUDA
  # do: Foundation.framework/Metal.framework/MetalKit.framework are
  # part of the OS itself and simply do not exist outside Apple
  # platforms (third_party/llama.cpp/ggml/src/ggml-metal/CMakeLists.txt's
  # own find_library(... REQUIRED) calls would otherwise be the first,
  # generic, MEMBRANE-unaware failure a non-Apple caller hits). Checked
  # via CMake's own APPLE variable (true only when the target OS is
  # Darwin), not a runtime probe -- there is no toolkit version or SDK
  # path that fixes this on a non-Apple OS. GGML_METAL is llama.cpp's
  # own option name (not defined by this file), read here only to
  # decide whether to run this pre-check.
  if(GGML_METAL AND NOT APPLE)
    message(FATAL_ERROR
      "Metal backend requested (-DGGML_METAL=ON) but the target OS is "
      "not Apple/Darwin. Metal is an Apple-only graphics/compute API "
      "(Foundation.framework/Metal.framework/MetalKit.framework do not "
      "exist on Linux/Windows) -- see docs/macos-metal.md.")
  endif()

  set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
  add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)

  execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/llama.cpp
    OUTPUT_VARIABLE MEMBRANE_LLAMA_CPP_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  configure_file(
    ${CMAKE_SOURCE_DIR}/cmake/llama_commit.h.in
    ${CMAKE_BINARY_DIR}/generated/membrane/llama_commit.h)

  # Phase 4.4: bit-exact adapter over ggml's own Q8_0/Q4_0 quantize and
  # dequantize implementations (see src/quant/ggml_quant.c and
  # docs/phase4-ggml-quant-parity.md). Only buildable here, since it
  # links directly against ggml -- unlike membrane_core, which stays
  # dependency-free for the portable Release/ASan suites.
  add_library(membrane_ggml_quant STATIC src/quant/ggml_quant.c)
  target_include_directories(membrane_ggml_quant PUBLIC include)
  target_include_directories(membrane_ggml_quant PRIVATE
    third_party/llama.cpp/ggml/include
    third_party/llama.cpp/ggml/src
    third_party/llama.cpp/ggml/src/ggml-cpu)
  target_link_libraries(membrane_ggml_quant PRIVATE ggml)

  if(MEMBRANE_BUILD_TESTS)
    # Phase 4.4 bit-parity tests: links ggml directly to make an
    # independent comparison call, so this can only build here (unlike
    # test_checkpoint, which is llama-free and built unconditionally
    # further down).
    add_executable(test_ggml_quant_parity
      tests/unit/test_ggml_quant_parity.c)
    target_link_libraries(test_ggml_quant_parity PRIVATE
      membrane_ggml_quant ggml)
    target_include_directories(test_ggml_quant_parity PRIVATE
      tests/unit
      third_party/llama.cpp/ggml/include
      third_party/llama.cpp/ggml/src
      third_party/llama.cpp/ggml/src/ggml-cpu)
    enable_testing()
    add_test(NAME test_ggml_quant_parity COMMAND test_ggml_quant_parity)

    # Phase 5.1: every membrane_simd_* backend (portable, in
    # membrane_core, no llama dependency) vs. membrane_ggml_quant's
    # oracle. Registered here (not with the portable tests further
    # down) only because it needs membrane_ggml_quant to reach that
    # oracle -- membrane_core/quant_simd.c itself has no llama
    # dependency and is built and usable even when
    # MEMBRANE_ENABLE_LLAMA is OFF.
    add_executable(test_quant_simd_parity
      tests/unit/test_quant_simd_parity.c)
    target_link_libraries(test_quant_simd_parity PRIVATE membrane_core
      membrane_ggml_quant)
    target_include_directories(test_quant_simd_parity PRIVATE tests/unit)
    add_test(NAME test_quant_simd_parity COMMAND test_quant_simd_parity)
  endif()

  add_subdirectory(tools/membrane-kv-capture)
  add_subdirectory(tools/membrane-kv-trace-capture)
  add_subdirectory(tools/membrane-kv-attn-trace-capture)
  add_subdirectory(tools/membrane-kv-attn-quality)
  add_subdirectory(tools/membrane-kv-quality)
  add_subdirectory(tools/membrane-kv-sensitivity)
  add_subdirectory(tools/membrane-policy-export)
  add_subdirectory(tools/membrane-kv-runtime)
  add_subdirectory(tools/membrane-kv-runtime-optimizer)
  add_subdirectory(tools/membrane-kv-variance)

  # Product Phase 5: the live llama.cpp shadow runtime (membrane-llama-
  # run). Deliberately does NOT use the kv_type_override patch above --
  # it observes live K/V projection tensors via the clean, unmodified,
  # publicly documented ggml_backend_sched_eval_callback
  # (llama_context_params.cb_eval), never replacing what llama.cpp
  # itself allocates/reads/writes for its KV cache. See
  # docs/live-runtime.md.
  add_subdirectory(tools/membrane-llama-runtime)

  # Product Phase 8: membrane-run, the user-facing entry point --
  # normal single-pass inference (default) and an explicit --compare-kv
  # mode reusing the Phase 7 3-pass comparison machinery via
  # membrane_decode_loop (tools/membrane-llama-runtime/decode_loop.h).
  # membrane-llama-run remains the diagnostic/legacy tool (Phase 5/6
  # shadow/inject modes).
  add_subdirectory(tools/membrane-run)

  # Mega Phase A, PR A2: `membrane`, the new product control CLI (model
  # registry today; `membrane serve`/`membrane status` are future work,
  # not this PR) -- a second binary alongside membrane-run, which stays
  # the unchanged inference entry point.
  add_subdirectory(tools/membrane)
endif()

if(MEMBRANE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/unit)

  # test_checkpoint.cpp exercises Phase 4.2's checkpoint/resume format
  # (tools/membrane-kv-runtime-optimizer/checkpoint.h), which has no
  # llama.cpp dependency -- built here, unconditionally, so it runs
  # under the standard Release/ASan suites even when
  # MEMBRANE_ENABLE_LLAMA is OFF, not only in llama-enabled builds.
  enable_language(CXX)
  add_executable(test_checkpoint
    tools/membrane-kv-runtime-optimizer/test_checkpoint.cpp)
  set_target_properties(test_checkpoint PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)
  target_link_libraries(test_checkpoint PRIVATE membrane_core)
  target_include_directories(test_checkpoint PRIVATE
    tools/membrane-kv-runtime-optimizer)
  add_test(NAME test_checkpoint COMMAND test_checkpoint)

  # Product Phase 1: membrane-demo's core pipeline logic, tested
  # directly against membrane_demo_core rather than only shelling out
  # to the CLI (see tools/membrane-demo/demo_core.h).
  add_executable(test_membrane_demo
    tools/membrane-demo/test_membrane_demo.c)
  target_link_libraries(test_membrane_demo PRIVATE membrane_demo_core)
  target_include_directories(test_membrane_demo PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/unit)
  add_test(NAME test_membrane_demo COMMAND test_membrane_demo)

  # CLI-level coverage of membrane-demo's argument validation and exit
  # codes, complementing the direct core-logic tests above.
  add_test(NAME demo_cli_default_exits_0 COMMAND membrane-demo)
  add_test(NAME demo_cli_json_exits_0
    COMMAND membrane-demo --blocks 256 --seed 7 --json)
  add_test(NAME demo_cli_rejects_zero_blocks
    COMMAND membrane-demo --blocks 0)
  set_tests_properties(demo_cli_rejects_zero_blocks PROPERTIES WILL_FAIL TRUE)
  add_test(NAME demo_cli_rejects_nonnumeric_blocks
    COMMAND membrane-demo --blocks abc)
  set_tests_properties(demo_cli_rejects_nonnumeric_blocks
    PROPERTIES WILL_FAIL TRUE)
  add_test(NAME demo_cli_rejects_oversized_blocks
    COMMAND membrane-demo --blocks 999999999999)
  set_tests_properties(demo_cli_rejects_oversized_blocks
    PROPERTIES WILL_FAIL TRUE)
  add_test(NAME demo_cli_rejects_unknown_option
    COMMAND membrane-demo --not-a-real-option)
  set_tests_properties(demo_cli_rejects_unknown_option
    PROPERTIES WILL_FAIL TRUE)

  # Product Phase 2: the shared generator/policy core, tested directly
  # (see tools/membrane-workload-core).
  add_executable(test_workload_core
    tools/membrane-workload-core/test_workload_core.c)
  target_link_libraries(test_workload_core PRIVATE membrane_workload_core)
  target_include_directories(test_workload_core PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/unit)
  add_test(NAME test_workload_core COMMAND test_workload_core)

  # Product Phase 3: the .memkv trace format reader/writer, including
  # adversarial/malformed-file handling (see
  # tools/membrane-workload-core/trace_format.h).
  add_executable(test_trace_format
    tools/membrane-workload-core/test_trace_format.c)
  target_link_libraries(test_trace_format PRIVATE membrane_workload_core)
  target_include_directories(test_trace_format PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/unit)
  add_test(NAME test_trace_format COMMAND test_trace_format)

  # membrane-quant-policy-bench's own core (config, matrix runner,
  # timing, printers), tested directly against membrane_bench_core.
  add_executable(test_membrane_bench
    tools/membrane-quant-policy-bench/test_membrane_bench.c)
  target_link_libraries(test_membrane_bench PRIVATE membrane_bench_core)
  target_include_directories(test_membrane_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/unit)
  add_test(NAME test_membrane_bench COMMAND test_membrane_bench)

  # CLI-level coverage of membrane-quant-policy-bench's exit codes,
  # complementing the direct core-logic tests above.
  add_test(NAME bench_cli_default_exits_0
    COMMAND membrane-quant-policy-bench --blocks 256 --iterations 2)
  add_test(NAME bench_cli_matrix_exits_0
    COMMAND membrane-quant-policy-bench --matrix --blocks 128 --iterations 2
      --json)
  add_test(NAME bench_cli_rejects_zero_blocks
    COMMAND membrane-quant-policy-bench --blocks 0)
  set_tests_properties(bench_cli_rejects_zero_blocks PROPERTIES
    WILL_FAIL TRUE)
  add_test(NAME bench_cli_rejects_zero_iterations
    COMMAND membrane-quant-policy-bench --iterations 0)
  set_tests_properties(bench_cli_rejects_zero_iterations PROPERTIES
    WILL_FAIL TRUE)
  add_test(NAME bench_cli_rejects_bad_policy
    COMMAND membrane-quant-policy-bench --policy bogus)
  set_tests_properties(bench_cli_rejects_bad_policy PROPERTIES
    WILL_FAIL TRUE)
  add_test(NAME bench_cli_rejects_bad_workload
    COMMAND membrane-quant-policy-bench --workload bogus)
  set_tests_properties(bench_cli_rejects_bad_workload PROPERTIES
    WILL_FAIL TRUE)
  add_test(NAME bench_cli_rejects_oversized_blocks
    COMMAND membrane-quant-policy-bench --blocks 999999999999)
  set_tests_properties(bench_cli_rejects_oversized_blocks PROPERTIES
    WILL_FAIL TRUE)

  # Product Phase 3: --trace CLI-level coverage, complementing the
  # direct membrane_bench_run_one/parse_args trace tests above.
  add_test(NAME bench_cli_rejects_missing_trace_file
    COMMAND membrane-quant-policy-bench --trace /no/such/file.memkv)
  set_tests_properties(bench_cli_rejects_missing_trace_file PROPERTIES
    WILL_FAIL TRUE)
  add_test(NAME bench_cli_rejects_trace_with_workload
    COMMAND membrane-quant-policy-bench --trace /no/such/file.memkv
      --workload synthetic-mixed)
  set_tests_properties(bench_cli_rejects_trace_with_workload PROPERTIES
    WILL_FAIL TRUE)
  add_test(NAME trace_export_cli_rejects_missing_args
    COMMAND membrane-kv-trace-export)
  set_tests_properties(trace_export_cli_rejects_missing_args PROPERTIES
    WILL_FAIL TRUE)

  # Product Phase 4: membrane-kv-trace-export's batch (.kvdump -> many
  # .memkv) core, tested directly against membrane_export_core.
  add_executable(test_export_core
    tools/membrane-kv-trace-export/test_export_core.c)
  target_link_libraries(test_export_core PRIVATE membrane_export_core)
  target_include_directories(test_export_core PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/unit)
  add_test(NAME test_export_core COMMAND test_export_core)

  add_test(NAME trace_export_cli_rejects_output_and_output_dir_together
    COMMAND membrane-kv-trace-export --input /no/such/file.kvdump
      --output /tmp/x.memkv --output-dir /tmp)
  set_tests_properties(
    trace_export_cli_rejects_output_and_output_dir_together PROPERTIES
    WILL_FAIL TRUE)
  add_test(NAME trace_export_cli_rejects_layer_with_output_dir
    COMMAND membrane-kv-trace-export --input /no/such/file.kvdump
      --output-dir /tmp --layer 0)
  set_tests_properties(trace_export_cli_rejects_layer_with_output_dir
    PROPERTIES WILL_FAIL TRUE)
  add_test(NAME trace_export_cli_rejects_missing_output_dir
    COMMAND membrane-kv-trace-export --input /no/such/file.kvdump
      --output-dir /no/such/directory/at/all)
  set_tests_properties(trace_export_cli_rejects_missing_output_dir
    PROPERTIES WILL_FAIL TRUE)

  # Product Phase 4: membrane-quant-policy-bench's trace-SET core
  # (discovery, aggregation, printers), tested directly against
  # membrane_bench_core.
  add_executable(test_trace_set
    tools/membrane-quant-policy-bench/test_trace_set.c)
  target_link_libraries(test_trace_set PRIVATE membrane_bench_core)
  target_include_directories(test_trace_set PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/unit)
  add_test(NAME test_trace_set COMMAND test_trace_set)

  add_test(NAME bench_cli_rejects_trace_dir_without_matrix
    COMMAND membrane-quant-policy-bench --trace-dir /tmp)
  set_tests_properties(bench_cli_rejects_trace_dir_without_matrix PROPERTIES
    WILL_FAIL TRUE)
  add_test(NAME bench_cli_rejects_trace_with_trace_dir
    COMMAND membrane-quant-policy-bench --trace /no/such/file.memkv
      --trace-dir /tmp --matrix)
  set_tests_properties(bench_cli_rejects_trace_with_trace_dir PROPERTIES
    WILL_FAIL TRUE)
  add_test(NAME bench_cli_rejects_missing_trace_dir
    COMMAND membrane-quant-policy-bench --trace-dir /no/such/dir --matrix)
  set_tests_properties(bench_cli_rejects_missing_trace_dir PROPERTIES
    WILL_FAIL TRUE)

  # Product Phase 5: the live shadow runtime's llama-free core (block
  # packing, telemetry, token-sequence comparison), tested directly --
  # no MEMBRANE_ENABLE_LLAMA, no model, no llama.cpp type anywhere in
  # this test or in membrane_runtime_core itself.
  add_executable(test_runtime_core
    tools/membrane-llama-runtime/test_runtime_core.c)
  target_link_libraries(test_runtime_core PRIVATE membrane_runtime_core)
  target_include_directories(test_runtime_core PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/unit)
  add_test(NAME test_runtime_core COMMAND test_runtime_core)

  # membrane-llama-run's CLI parser (cli_parse.h/.cpp), exercised through
  # the real parse_opts()/argv boundary -- no model load, no llama.cpp
  # dependency. Covers the strict numeric parsing fix (rejecting a
  # leading '-' that strtoull() alone would silently wrap) and the
  # existing scope/mode validation.
  add_executable(test_cli_parse
    tools/membrane-llama-runtime/test_cli_parse.cpp)
  set_target_properties(test_cli_parse PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)
  target_link_libraries(test_cli_parse PRIVATE membrane_cli_parse)
  target_include_directories(test_cli_parse PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/unit)
  add_test(NAME test_cli_parse COMMAND test_cli_parse)

  # Product Phase 7: RSS-checkpoint and KV byte-accounting math, tested
  # directly -- no MEMBRANE_ENABLE_LLAMA, no model, no llama.cpp type
  # anywhere in this test or in membrane_kv_store_telemetry itself.
  add_executable(test_kv_store_telemetry
    tools/membrane-llama-runtime/test_kv_store_telemetry.c)
  target_link_libraries(test_kv_store_telemetry PRIVATE
    membrane_kv_store_telemetry)
  target_include_directories(test_kv_store_telemetry PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/unit)
  add_test(NAME test_kv_store_telemetry COMMAND test_kv_store_telemetry)

  # Product Phase 8: membrane-run's product CLI parser and
  # compatibility pre-check, tested directly -- no MEMBRANE_ENABLE_LLAMA,
  # no model, no llama.cpp type anywhere in either library or its test.
  add_executable(test_product_cli
    tools/membrane-run/test_product_cli.cpp)
  set_target_properties(test_product_cli PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)
  target_link_libraries(test_product_cli PRIVATE membrane_product_cli)
  target_include_directories(test_product_cli PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/unit)
  add_test(NAME test_product_cli COMMAND test_product_cli)

  add_executable(test_compat_check
    tools/membrane-run/test_compat_check.c)
  target_link_libraries(test_compat_check PRIVATE membrane_compat_check)
  target_include_directories(test_compat_check PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/unit)
  add_test(NAME test_compat_check COMMAND test_compat_check)

  add_executable(test_gpu_policy
    tools/membrane-run/test_gpu_policy.c)
  target_link_libraries(test_gpu_policy PRIVATE membrane_gpu_policy)
  target_include_directories(test_gpu_policy PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/unit)
  add_test(NAME test_gpu_policy COMMAND test_gpu_policy)

  # Phase 12H: --kv-placement's pure planner tests.
  add_executable(test_kv_residency_policy
    tools/membrane-run/test_kv_residency_policy.c)
  target_link_libraries(test_kv_residency_policy PRIVATE
    membrane_kv_residency_policy)
  target_include_directories(test_kv_residency_policy PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/unit)
  add_test(NAME test_kv_residency_policy COMMAND test_kv_residency_policy)

  add_executable(test_adaptive_kv_policy
    tools/membrane-run/test_adaptive_kv_policy.c)
  target_link_libraries(test_adaptive_kv_policy PRIVATE
    membrane_adaptive_kv_policy)
  target_include_directories(test_adaptive_kv_policy PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/unit)
  add_test(NAME test_adaptive_kv_policy COMMAND test_adaptive_kv_policy)

  add_executable(test_joint_planner
    tools/membrane-run/test_joint_planner.c)
  target_link_libraries(test_joint_planner PRIVATE membrane_joint_planner)
  target_include_directories(test_joint_planner PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/unit)
  add_test(NAME test_joint_planner COMMAND test_joint_planner)

  # Phase 34: host-memory guard's own tests -- llama-free, pure
  # arithmetic, no model/GPU/proc access.
  add_executable(test_host_memory_guard
    tools/membrane-run/test_host_memory_guard.c)
  target_link_libraries(test_host_memory_guard PRIVATE
    membrane_host_memory_guard)
  target_include_directories(test_host_memory_guard PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/unit)
  add_test(NAME test_host_memory_guard COMMAND test_host_memory_guard)

  # Phase 33: safe context recommendation core's own tests -- llama-
  # free, synthetic model/device facts only (no model download, no
  # GPU), same pattern as test_joint_planner above.
  add_executable(test_context_recommender
    tools/membrane-run/test_context_recommender.c)
  target_link_libraries(test_context_recommender PRIVATE
    membrane_context_recommender)
  target_include_directories(test_context_recommender PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/unit)
  add_test(NAME test_context_recommender COMMAND test_context_recommender)

  # Phase 35: --ctx auto's own pure CLI-adapter tests -- llama-free.
  add_executable(test_context_auto_cli
    tools/membrane-run/test_context_auto_cli.c)
  target_link_libraries(test_context_auto_cli PRIVATE
    membrane_context_auto_cli)
  target_include_directories(test_context_auto_cli PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/unit)
  add_test(NAME test_context_auto_cli COMMAND test_context_auto_cli)

  # Phase 21: the bounded apply-time fallback controller's own tests --
  # llama-free (synthetic apply_fn/refresh_fn test doubles), plus a
  # handful of tests that run the REAL membrane_joint_plan_resolve()
  # (Section 20) to prove explicit-constraint preservation against real
  # planner output, not just a hand-built candidate array.
  add_executable(test_auto_fallback
    tools/membrane-run/test_auto_fallback.c)
  target_link_libraries(test_auto_fallback PRIVATE membrane_auto_fallback
    membrane_joint_planner)
  target_include_directories(test_auto_fallback PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/unit)
  add_test(NAME test_auto_fallback COMMAND test_auto_fallback)

  # Phase 6.2: hot-cache/prefetch and working-set-policy correctness
  # tests, plus an engine determinism check -- llama-free (they test
  # membrane_wssim_core's C++ logic directly, not real inference), so
  # built unconditionally like test_checkpoint above.
  add_executable(test_hotcache
    tools/membrane-kv-workingset-sim/test_hotcache.cpp)
  target_link_libraries(test_hotcache PRIVATE membrane_wssim_core)
  add_test(NAME test_hotcache COMMAND test_hotcache)

  add_executable(test_workingset_policy
    tools/membrane-kv-workingset-sim/test_workingset_policy.cpp)
  target_link_libraries(test_workingset_policy PRIVATE membrane_wssim_core)
  add_test(NAME test_workingset_policy COMMAND test_workingset_policy)

  # Phase 6.5: out-of-core streaming trace reader + bounded chunk
  # cache -- LRU/pin/duplicate-load-avoidance correctness and
  # in-memory/mmap/streaming backend parity, llama-free like the
  # tests above.
  #
  # Note for MEMBRANE_ENABLE_TSAN builds: on this machine's kernel,
  # ThreadSanitizer's shadow-memory layout can reject an otherwise
  # ordinary mmap() (or, empirically, ANY TSan-instrumented binary,
  # not just ones calling mmap directly) as an "unexpected memory
  # mapping" purely depending on where ASLR happens to place it -- a
  # known TSan/kernel interaction, not a real data race. Run
  # `ctest --test-dir build-tsan` itself under `setarch $(uname -m)
  # -R` (disables ASLR for the whole process tree) rather than
  # special-casing individual tests here.
  add_executable(test_attn_trace_reader
    tools/membrane-kv-workingset-sim/test_attn_trace_reader.cpp)
  target_link_libraries(test_attn_trace_reader PRIVATE membrane_wssim_core)
  add_test(NAME test_attn_trace_reader COMMAND test_attn_trace_reader)

  # Phase 6.3: concurrent exact-retrieval engine correctness tests
  # (device capacity enforcement, contention, micro-batching,
  # deterministic replay) -- hand-built fixtures, no real trace file
  # dependency, so built unconditionally like the tests above.
  add_executable(test_exact_engine
    tools/membrane-kv-exact-sim/test_exact_engine.cpp)
  target_link_libraries(test_exact_engine PRIVATE membrane_exactsim_core)
  add_test(NAME test_exact_engine COMMAND test_exact_engine)

  # Phase 6.5: calibrate() vs calibrate_streamed() (mmap/streaming
  # backends) parity.
  add_executable(test_calibrate_streamed
    tools/membrane-kv-exact-sim/test_calibrate_streamed.cpp)
  target_link_libraries(test_calibrate_streamed
    PRIVATE membrane_exactsim_core)
  add_test(NAME test_calibrate_streamed COMMAND test_calibrate_streamed)

  # Phase 6.5: bounded (spill-to-disk-if-needed) exact quantile
  # accumulator -- must match a plain std::sort reference exactly.
  add_executable(test_bounded_quantile
    tools/membrane-kv-exact-sim/test_bounded_quantile.cpp)
  target_link_libraries(test_bounded_quantile PRIVATE membrane_exactsim_core)
  add_test(NAME test_bounded_quantile COMMAND test_bounded_quantile)

  # Phase 6.5: mem_guard_t -- real /proc-based RSS/memory monitoring
  # and the escalating action it drives.
  add_executable(test_mem_guard
    tools/membrane-kv-exact-sim/test_mem_guard.cpp)
  target_link_libraries(test_mem_guard PRIVATE membrane_exactsim_core)
  add_test(NAME test_mem_guard COMMAND test_mem_guard)

  # Phase 6.5 item 14: end-to-end interrupted/resumed sweep +
  # corrupted-checkpoint rejection, against the REAL binaries and a
  # REAL (small) native trace already committed to the repo -- see
  # the script's own header for why this stays fast despite exercising
  # the real CLI end to end (it only ever waits out the two
  # analytical, non-calibrate() comparisons).
  find_program(MEMBRANE_BASH bash)
  if(MEMBRANE_BASH AND EXISTS
      ${CMAKE_SOURCE_DIR}/benchmarks/cxl-sim/traces/SmolLM2-135M-Instruct-f16-long.attntrace)
    add_test(NAME test_interrupted_resume
      COMMAND ${MEMBRANE_BASH}
        ${CMAKE_SOURCE_DIR}/tools/membrane-kv-exact-sim/test_interrupted_resume.sh
        $<TARGET_FILE:membrane-kv-exact-sim>
        $<TARGET_FILE:membrane-kv-exact-sim-verify>
        ${CMAKE_SOURCE_DIR}/benchmarks/cxl-sim/traces/SmolLM2-135M-Instruct-f16-long.attntrace
        ${CMAKE_SOURCE_DIR})
    # Bumped from 400s: the test now polls (up to 240s in phase 1, up
    # to 60s in phase 3) for a real completion signal instead of a
    # fixed sleep, after a fixed 45s sleep was found to produce 0
    # completed scenarios on GitHub's CI runner (a real runner-speed
    # gap, not just sanitizer overhead -- it reproduced under a plain
    # Debug build too). 700s covers both poll ceilings plus real
    # margin for trace generation/kill/resume/integrity-check overhead.
    set_tests_properties(test_interrupted_resume PROPERTIES TIMEOUT 700)
  endif()
endif()

# ---------------------------------------------------------------------
# Phase 29: .deb packaging (CPack). Configured unconditionally (cheap,
# and `cmake --build <dir> --target package` is opt-in -- nothing here
# runs on a normal configure/build) but only produces a non-empty
# package when MEMBRANE_ENABLE_LLAMA=ON, since
# tools/membrane-run/CMakeLists.txt's install() rules are the ONLY
# install() rules anywhere in this tree (verified directly: `grep -rn
# "^\s*install(" --include=CMakeLists.txt .` finds nothing else) -- the
# package can never accidentally pick up tests, benchmarks, research
# tools, or model files, because nothing else ever gets installed in
# the first place, packaged or not.
#
# Single source of truth for the package version: MEMBRANE_VERSION
# (tools/membrane-run/product_cli.h) is parsed here, not duplicated --
# a release version bump only ever needs to change that one #define
# (see product_cli.h's own header comment). file(STRINGS ...) reads the
# header as plain text (no compiler invocation needed), so this works
# identically whether or not MEMBRANE_ENABLE_LLAMA is set.
file(STRINGS ${CMAKE_SOURCE_DIR}/tools/membrane-run/product_cli.h
  MEMBRANE_VERSION_DEFINE_LINE REGEX "define MEMBRANE_VERSION")
string(REGEX MATCH "\"([^\"]+)\"" _MEMBRANE_VERSION_MATCH
  "${MEMBRANE_VERSION_DEFINE_LINE}")
set(MEMBRANE_PRODUCT_VERSION "${CMAKE_MATCH_1}")
if(NOT MEMBRANE_PRODUCT_VERSION)
  message(FATAL_ERROR
    "Could not parse MEMBRANE_VERSION from "
    "tools/membrane-run/product_cli.h -- packaging version would be "
    "empty/wrong, refusing to configure CPack with a guessed value")
endif()

# Debian version ordering (Debian Policy 5.6.12): a bare hyphen in an
# upstream_version is the debian_revision separator, and dpkg's version
# comparison algorithm sorts "-rc3" AFTER no suffix at all -- exactly
# backwards from "rc3 is a prerelease of the final version" (0.3.0-rc3
# would compare as NEWER than 0.3.0, so an apt upgrade from the rc to
# the real 0.3.0 release would look like a downgrade and be refused).
# "~" is the documented convention for this: it sorts before
# everything, including the empty string, so "0.3.0~rc3" correctly
# orders BEFORE "0.3.0". Only ever touches a "-rcN"/"-betaN"/etc.
# suffix if MEMBRANE_PRODUCT_VERSION has one -- a plain stable version
# like "0.2.0" has no hyphen and passes through unchanged. The product
# itself keeps reporting the unmapped MEMBRANE_VERSION verbatim
# (`membrane-run --version` prints "MEMBRANE 0.3.0-rc3", unaffected by
# this) -- this mapping exists ONLY for the .deb's own Version: field,
# documented in docs/install.md's packaging section.
string(REPLACE "-" "~" MEMBRANE_DEBIAN_VERSION "${MEMBRANE_PRODUCT_VERSION}")

# Phase 30, Section 2-4/16: the Vulkan-enabled build is the ONE
# official Linux amd64 .deb -- proven (real container validation,
# Phase 29 and Phase 30) to run correctly CPU-only, handle a host with
# libvulkan1 but zero visible GPU devices cleanly (--doctor gives an
# accurate combined warning, --list-devices shows CPU only, a real
# generation run succeeds), and never touch a GPU unless --auto or an
# explicit --gpu-layers/--kv-placement flag asks for one -- see
# results/distribution/validation.json and results/first-run/
# validation.json. Its ONLY extra cost over the CPU-only build is one
# additional real runtime dependency (libvulkan1, auto-detected below,
# never hand-maintained) -- so it is the simplest honest choice
# (Section 2: "Prefer the simplest honest user experience"): one
# package name, "membrane", for the official artifact, keeping the
# CPU-only variant (still built and validated every CI run --
# packaging-smoke) as "membrane-cpu" -- a CI-validation/build-your-own
# artifact, explicitly NOT the official release asset (see
# docs/install.md and results/release-artifacts/manifest.json for the
# full policy). This is a build-time choice (whether -DGGML_VULKAN=ON
# was passed), not a separate CMake option of its own -- there is
# nothing else that distinguishes the two variants.
if(GGML_VULKAN)
  set(CPACK_PACKAGE_NAME "membrane")
else()
  set(CPACK_PACKAGE_NAME "membrane-cpu")
endif()
set(CPACK_PACKAGE_VERSION "${MEMBRANE_DEBIAN_VERSION}")
set(CPACK_PACKAGE_VENDOR "Kadir Eren Altintas")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
  "Memory-aware KV-cache planning runtime for llama.cpp")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/kadireren7/membrane")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_CONTACT
  "Kadir Eren Altintas <kadirerenaltintas072@gmail.com>")
# Debian packages install under /usr, never /usr/local -- this only
# changes where CPack STAGES files for the .deb's own payload; the
# ordinary `cmake --install --prefix ...` workflow (docs/install.md)
# is completely separate and unaffected by this variable.
set(CPACK_PACKAGING_INSTALL_PREFIX "/usr")
set(CPACK_STRIP_FILES ON)

set(CPACK_GENERATOR "DEB")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER
  "Kadir Eren Altintas <kadirerenaltintas072@gmail.com>")
set(CPACK_DEBIAN_PACKAGE_SECTION "utils")
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "${CPACK_PACKAGE_HOMEPAGE_URL}")
if(GGML_VULKAN)
  set(MEMBRANE_CPACK_BACKEND_DESC
" Built with Vulkan GPU offload support (requires libvulkan1 at
 runtime, declared as a real package dependency below) -- runs
 CPU-only by default; GPU offload is opt-in via --auto or an
 explicit --gpu-layers/--kv-placement flag, and this package works
 correctly with no GPU visible at all (CPU fallback is not a
 separate build).")
else()
  set(MEMBRANE_CPACK_BACKEND_DESC
" CPU-only build: no Vulkan/GPU support compiled in. This is a CI-
 validation/build-your-own artifact, not the official release
 package -- see https://github.com/kadireren7/membrane's
 docs/install.md. Prefer the membrane package (Vulkan-enabled, still
 works CPU-only) for a normal install.")
endif()
set(CPACK_DEBIAN_PACKAGE_DESCRIPTION
"${CPACK_PACKAGE_DESCRIPTION_SUMMARY}
 Resolves GPU offload, KV precision, and CPU/GPU KV residency before
 generation, with an inspectable plan and machine-readable
 diagnostics, on top of an unmodified ggml/llama.cpp inference path.
 .
 Provides the membrane-run command-line tool. This package does not
 include any model files -- point membrane-run at your own local
 .gguf file.
 .
${MEMBRANE_CPACK_BACKEND_DESC}
 .
 Debian-family Linux, amd64 only. See
 https://github.com/kadireren7/membrane for details.")
# Real, auto-detected runtime dependencies (dpkg-shlibdeps against the
# actual installed ELF binaries), not a hand-maintained guess -- a
# CPU-only build (no -DGGML_VULKAN=ON) never declares a Vulkan
# dependency, and a Vulkan build declares exactly what libggml-vulkan
# actually links against (e.g. libvulkan.so.1) -- whichever this exact
# build produces. glslc (a build-time-only shader compiler, never
# linked into the installed binary) can never appear here for that
# same reason: dpkg-shlibdeps only inspects real ELF NEEDED entries.
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
# membrane-run's own installed libllama.so/libggml*.so (tools/membrane-
# run/CMakeLists.txt's install loop) live inside THIS package, not a
# separate system package -- without this, dpkg-shlibdeps has no way
# to resolve those NEEDED entries against an apt-installed package and
# either errors or (worse) silently omits the dependency line entirely.
# CMake 3.20+ only; guarded so a configure on an older (but still
# >=3.16, this project's own minimum) CMake still produces a package,
# just without this refinement -- dpkg-shlibdeps would then warn on
# stderr about unresolved libllama.so/libggml*.so instead of failing.
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.20")
  set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS_PRIVATE_DIRS
    "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
endif()
set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)

include(CPack)
