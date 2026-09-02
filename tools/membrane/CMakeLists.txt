# Mega Phase A, PR A2: the local model registry + the new `membrane`
# control CLI. Only evaluated when MEMBRANE_ENABLE_LLAMA=ON (see the
# top-level CMakeLists.txt's if(MEMBRANE_ENABLE_LLAMA) block) -- like
# membrane_gpu_device/membrane_decode_loop, this needs real GGUF
# validation (gpu_device.h's membrane_gpu_estimate_model()) and its one
# real dependency, nlohmann::json, is vendored inside the llama.cpp
# submodule (third_party/llama.cpp/vendor/nlohmann) -- see registry_core.h's
# own top comment for why this module is llama-free in spirit even though
# it is built here.
set(MEMBRANE_NLOHMANN_JSON_INCLUDE_DIR
  ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/vendor)
# Mega Phase A, PR A3: cpp-httplib -- also vendored inside the llama.cpp
# submodule (same directory as nlohmann/json above), so `membrane serve`
# adds zero new external dependencies (Section 17 of the task: "do not
# import a huge web framework... casually"). Header-only, HTTP only (no
# CPPHTTPLIB_OPENSSL_SUPPORT defined -- this server is loopback-only by
# default and never needs TLS), so no extra system library beyond
# pthread (already required by std::thread, used for the listener
# thread).
set(MEMBRANE_CPP_HTTPLIB_INCLUDE_DIR
  ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/vendor/cpp-httplib)
find_package(Threads REQUIRED)

# This vendored copy is split header+impl (httplib.h declares, httplib.cpp
# defines -- confirmed directly: linking against the header alone produced
# undefined references to every httplib::Server/Response member function),
# not the classic single-header drop-in -- compiled once here as its own
# static library, upstream's own -w (this is third-party code, not ours)
# carried over from its own vendor/cpp-httplib/CMakeLists.txt.
add_library(membrane_cpp_httplib STATIC
  ${MEMBRANE_CPP_HTTPLIB_INCLUDE_DIR}/httplib.cpp)
set_target_properties(membrane_cpp_httplib PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON)
target_compile_options(membrane_cpp_httplib PRIVATE -w)
target_include_directories(membrane_cpp_httplib PUBLIC
  ${MEMBRANE_CPP_HTTPLIB_INCLUDE_DIR})
target_link_libraries(membrane_cpp_httplib PUBLIC Threads::Threads)

add_library(membrane_registry_core STATIC registry_core.cpp)
set_target_properties(membrane_registry_core PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)
target_include_directories(membrane_registry_core PUBLIC
  ${CMAKE_SOURCE_DIR}/tools/membrane)
target_include_directories(membrane_registry_core PRIVATE
  ${MEMBRANE_NLOHMANN_JSON_INCLUDE_DIR})

# Mega Phase A, PR A3: the local OpenAI-compatible HTTP server (server.h's
# own top comment has the full contract) -- its own library, not compiled
# directly into the `membrane` executable, so test_server.cpp (below) can
# link and drive a REAL membrane_server_run() instance without duplicating
# any of its code.
add_library(membrane_server STATIC server.cpp)
set_target_properties(membrane_server PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)
target_link_libraries(membrane_server PUBLIC membrane_registry_core
  membrane_runtime_session membrane_cpp_httplib membrane_product_cli llama)
target_include_directories(membrane_server PUBLIC
  ${CMAKE_SOURCE_DIR}/tools/membrane)
target_include_directories(membrane_server PRIVATE
  ${CMAKE_SOURCE_DIR}/tools/membrane-run
  ${CMAKE_SOURCE_DIR}/tools/membrane-llama-runtime
  ${MEMBRANE_NLOHMANN_JSON_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include)

add_executable(membrane main.cpp model_cmd.cpp)
set_target_properties(membrane PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)
target_link_libraries(membrane PRIVATE membrane_registry_core
  membrane_gpu_device membrane_runtime_core membrane_product_cli
  membrane_runtime_session membrane_server llama)
target_include_directories(membrane PRIVATE
  ${CMAKE_SOURCE_DIR}/tools/membrane-run
  ${CMAKE_SOURCE_DIR}/tools/membrane-llama-runtime
  ${MEMBRANE_NLOHMANN_JSON_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include)

# Same install treatment as membrane-run (docs/install.md, packaging.md):
# a second product binary in the same package, same RPATH-to-libdir
# convention so it works outside the build tree without LD_LIBRARY_PATH.
file(RELATIVE_PATH MEMBRANE_RPATH_TO_LIBDIR
  "${CMAKE_INSTALL_FULL_BINDIR}" "${CMAKE_INSTALL_FULL_LIBDIR}")
set_target_properties(membrane PROPERTIES
  INSTALL_RPATH "$ORIGIN/${MEMBRANE_RPATH_TO_LIBDIR}"
  BUILD_WITH_INSTALL_RPATH FALSE
  SKIP_BUILD_RPATH FALSE)
install(TARGETS membrane RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(MEMBRANE_BUILD_TESTS)
  # CI-safe: exercises registry_core.h's pure add/remove/list/find/
  # identity-check logic against a temp directory this test creates and
  # removes itself -- never the user's real registry, no real GGUF model
  # needed (registry_core itself never validates GGUF-ness; that is
  # model_cmd.cpp's job, covered separately below).
  add_executable(test_registry_core test_registry_core.cpp)
  set_target_properties(test_registry_core PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)
  target_link_libraries(test_registry_core PRIVATE membrane_registry_core)
  target_include_directories(test_registry_core PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/unit)
  add_test(NAME test_registry_core COMMAND test_registry_core)

  # Mega Phase A, PR A3: black-box HTTP tests (test_server.cpp's own top
  # comment) against a real membrane_server_run() instance -- CI-safe, no
  # real GGUF model (an isolated, empty temp-directory registry).
  add_executable(test_server test_server.cpp)
  set_target_properties(test_server PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)
  target_link_libraries(test_server PRIVATE membrane_server llama)
  target_include_directories(test_server PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/unit
    ${MEMBRANE_NLOHMANN_JSON_INCLUDE_DIR})
  add_test(NAME test_server COMMAND test_server)
endif()
