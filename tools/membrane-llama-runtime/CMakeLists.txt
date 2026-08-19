# Only evaluated when MEMBRANE_ENABLE_LLAMA=ON (see the top-level
# CMakeLists.txt's if(MEMBRANE_ENABLE_LLAMA) block). membrane_runtime_core
# (runtime_core.c, the llama-free MEMBRANE-side logic) is registered
# unconditionally in the top-level CMakeLists.txt instead, so it and its
# test build under the default llama-free configuration too.
# Shared llama-enabled decode-loop primitives (model-agnostic prompt/
# generation loop, Phase 7 kv-store pass) -- both membrane-llama-run
# (legacy/diagnostic) and membrane-run (product entry point,
# tools/membrane-run) link this instead of each defining their own
# copy. Needs llama/ggml types, so unlike membrane_runtime_core it can
# only be built here, inside the MEMBRANE_ENABLE_LLAMA block.
add_library(membrane_decode_loop STATIC decode_loop.cpp llama_hook.cpp)
set_target_properties(membrane_decode_loop PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)
target_link_libraries(membrane_decode_loop PUBLIC membrane_runtime_core
  membrane_kv_store_telemetry llama)
target_include_directories(membrane_decode_loop PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include)

if(MEMBRANE_BUILD_TESTS)
  # Phase 12H review fix: kv_placement_dev_override_cb needs real
  # ggml_backend_dev_by_type()/ggml_backend_dev_t, so unlike the
  # llama-free planner tests (test_kv_residency_policy), this can only
  # build here, inside MEMBRANE_ENABLE_LLAMA.
  add_executable(test_decode_loop test_decode_loop.cpp)
  set_target_properties(test_decode_loop PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)
  target_link_libraries(test_decode_loop PRIVATE membrane_decode_loop llama)
  target_include_directories(test_decode_loop PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/unit
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include)
  add_test(NAME test_decode_loop COMMAND test_decode_loop)
endif()

add_executable(membrane-llama-run main.cpp)
set_target_properties(membrane-llama-run PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)
target_link_libraries(membrane-llama-run PRIVATE membrane_runtime_core
  membrane_cli_parse membrane_kv_store_telemetry membrane_decode_loop llama)
target_include_directories(membrane-llama-run PRIVATE
  ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include)
