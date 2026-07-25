cmake_minimum_required(VERSION 3.16)
project(membrane C)

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

add_library(membrane_core
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
)
target_include_directories(membrane_core PUBLIC include)
target_link_libraries(membrane_core PUBLIC membrane_sanitizers Threads::Threads m)

add_subdirectory(tools/membrane-bench)
add_subdirectory(tools/membrane-store-bench)
add_subdirectory(tools/membrane-kv-analyze)
add_subdirectory(tools/membrane-quant-bench)
add_subdirectory(tools/membrane-hw-sim)

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
  set(MEMBRANE_LLAMA_PATCH
    ${CMAKE_SOURCE_DIR}/patches/llama.cpp-membrane-kv-type-override.patch)
  execute_process(
    COMMAND git apply --reverse --check ${MEMBRANE_LLAMA_PATCH}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/llama.cpp
    RESULT_VARIABLE MEMBRANE_PATCH_ALREADY_APPLIED
    OUTPUT_QUIET ERROR_QUIET)
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
  add_subdirectory(tools/membrane-kv-quality)
  add_subdirectory(tools/membrane-kv-sensitivity)
  add_subdirectory(tools/membrane-policy-export)
  add_subdirectory(tools/membrane-kv-runtime)
  add_subdirectory(tools/membrane-kv-runtime-optimizer)
  add_subdirectory(tools/membrane-kv-variance)
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
endif()
