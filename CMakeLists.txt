cmake_minimum_required(VERSION 3.16)
project(membrane C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(MEMBRANE_ENABLE_SANITIZERS
    "Enable AddressSanitizer and UndefinedBehaviorSanitizer"
    OFF)
option(MEMBRANE_ENABLE_ASAN
    "Deprecated alias for MEMBRANE_ENABLE_SANITIZERS"
    OFF)
option(MEMBRANE_BUILD_TESTS "Build unit tests" ON)

if(MEMBRANE_ENABLE_ASAN)
  message(DEPRECATION
    "MEMBRANE_ENABLE_ASAN is deprecated; use MEMBRANE_ENABLE_SANITIZERS")
  set(MEMBRANE_ENABLE_SANITIZERS ON)
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

add_library(membrane_core
  src/block/block.c
  src/codecs/raw.c
  src/codecs/rle.c
  src/codecs/codec_registry.c
  src/stats.c
)
target_include_directories(membrane_core PUBLIC include)
target_link_libraries(membrane_core PUBLIC membrane_sanitizers)

add_subdirectory(tools/membrane-bench)

if(MEMBRANE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/unit)
endif()
