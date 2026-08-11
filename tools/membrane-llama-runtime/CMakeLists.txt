# Only evaluated when MEMBRANE_ENABLE_LLAMA=ON (see the top-level
# CMakeLists.txt's if(MEMBRANE_ENABLE_LLAMA) block). membrane_runtime_core
# (runtime_core.c, the llama-free MEMBRANE-side logic) is registered
# unconditionally in the top-level CMakeLists.txt instead, so it and its
# test build under the default llama-free configuration too.
add_executable(membrane-llama-run main.cpp llama_hook.cpp)
set_target_properties(membrane-llama-run PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)
target_link_libraries(membrane-llama-run PRIVATE membrane_runtime_core
  membrane_cli_parse llama)
target_include_directories(membrane-llama-run PRIVATE
  ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include)
