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

# Mega Phase B, PR B1: the shared "mkdir -p + atomic write" helper (fs_util.h's
# own top comment) -- registry_core.cpp, server_config.cpp, and
# service_cmd.cpp's unit-file write all link this instead of each carrying
# their own copy.
add_library(membrane_fs_util STATIC fs_util.cpp)
set_target_properties(membrane_fs_util PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)
target_link_libraries(membrane_fs_util PUBLIC membrane_sanitizers)
target_include_directories(membrane_fs_util PUBLIC
  ${CMAKE_SOURCE_DIR}/tools/membrane)

# Mega Phase B, PR B1: minimal safe subprocess runner (subprocess.h's own
# top comment) -- fork()+execvp(), never system()/popen().
add_library(membrane_subprocess STATIC subprocess.cpp)
set_target_properties(membrane_subprocess PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)
target_link_libraries(membrane_subprocess PUBLIC membrane_sanitizers)
target_include_directories(membrane_subprocess PUBLIC
  ${CMAKE_SOURCE_DIR}/tools/membrane)

# Mega Phase B, PR B1: pure systemd --user unit-file generation (systemd_
# unit.h's own top comment) -- no filesystem/systemctl access here.
add_library(membrane_systemd_unit STATIC systemd_unit.cpp)
set_target_properties(membrane_systemd_unit PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)
target_link_libraries(membrane_systemd_unit PUBLIC membrane_sanitizers)
target_include_directories(membrane_systemd_unit PUBLIC
  ${CMAKE_SOURCE_DIR}/tools/membrane)

# Mega Phase B, PR B1: the persistent `membrane serve` config (server_
# config.h's own top comment) -- listen_address/port/default_model/
# log_level, atomically written via membrane_fs_util.
add_library(membrane_server_config STATIC server_config.cpp)
set_target_properties(membrane_server_config PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)
target_link_libraries(membrane_server_config PUBLIC membrane_sanitizers
  membrane_fs_util)
target_include_directories(membrane_server_config PUBLIC
  ${CMAKE_SOURCE_DIR}/tools/membrane)
target_include_directories(membrane_server_config PRIVATE
  ${MEMBRANE_NLOHMANN_JSON_INCLUDE_DIR})

add_library(membrane_registry_core STATIC registry_core.cpp)
set_target_properties(membrane_registry_core PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)
# Real bug found and fixed (Mega Phase A, PR A3): this library previously
# linked nothing at all, so it was NEVER actually compiled with ASan/
# UBSan/TSan instrumentation under a sanitizer configure -- confirmed
# directly (`nm` on the built .a/test binary showed zero __asan_*
# symbols) -- meaning PR A2's own "ASan/UBSan clean" claim for
# test_registry_core was not actually testing anything. membrane_
# sanitizers is the project's own INTERFACE library carrying the real
# -fsanitize flags (see the root CMakeLists.txt's own top-of-file
# comment) -- every other pure module links it PUBLIC; this one now
# does too.
target_link_libraries(membrane_registry_core PUBLIC membrane_sanitizers
  membrane_fs_util)
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

# Mega Phase B, PR B1: a thin HTTP client against a running `membrane
# serve`/service instance's own GET /v1/status (status_client.h's own top
# comment) -- shared by `membrane status` (existing, A4) and `membrane
# service status` (new, this PR).
add_library(membrane_status_client STATIC status_client.cpp)
set_target_properties(membrane_status_client PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)
target_link_libraries(membrane_status_client PUBLIC membrane_sanitizers
  membrane_cpp_httplib)
target_include_directories(membrane_status_client PUBLIC
  ${CMAKE_SOURCE_DIR}/tools/membrane)
target_include_directories(membrane_status_client PRIVATE
  ${MEMBRANE_NLOHMANN_JSON_INCLUDE_DIR})

# Mega Phase B, PR B1: `membrane service ...` itself (service_cmd.h's own
# top comment) -- installs/manages membrane.service as a systemd --user
# unit; never a second daemon-supervision implementation.
add_library(membrane_service_cmd STATIC service_cmd.cpp)
set_target_properties(membrane_service_cmd PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)
target_link_libraries(membrane_service_cmd PUBLIC membrane_sanitizers
  membrane_systemd_unit membrane_server_config membrane_subprocess
  membrane_status_client membrane_fs_util membrane_product_cli)
target_include_directories(membrane_service_cmd PUBLIC
  ${CMAKE_SOURCE_DIR}/tools/membrane)
target_include_directories(membrane_service_cmd PRIVATE
  ${MEMBRANE_NLOHMANN_JSON_INCLUDE_DIR})

add_executable(membrane main.cpp model_cmd.cpp)
set_target_properties(membrane PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)
target_link_libraries(membrane PRIVATE membrane_registry_core
  membrane_gpu_device membrane_runtime_core membrane_product_cli
  membrane_runtime_session membrane_server membrane_service_cmd
  membrane_status_client membrane_server_config llama)
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

  # Mega Phase B, PR B1, Section 13: pure tests for the new service-
  # lifecycle modules -- unit generation/path handling, config
  # serialization, atomic write, and subprocess injection safety. None of
  # these touch a real systemd/journalctl (that is the real local smoke
  # test's job, run manually -- see docs/service.md), so they are CI-safe
  # and hermetic like every other test in this file.
  add_executable(test_systemd_unit test_systemd_unit.cpp)
  set_target_properties(test_systemd_unit PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)
  target_link_libraries(test_systemd_unit PRIVATE membrane_systemd_unit)
  target_include_directories(test_systemd_unit PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/unit)
  add_test(NAME test_systemd_unit COMMAND test_systemd_unit)

  add_executable(test_server_config test_server_config.cpp)
  set_target_properties(test_server_config PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)
  target_link_libraries(test_server_config PRIVATE membrane_server_config)
  target_include_directories(test_server_config PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/unit)
  add_test(NAME test_server_config COMMAND test_server_config)

  add_executable(test_fs_util test_fs_util.cpp)
  set_target_properties(test_fs_util PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)
  target_link_libraries(test_fs_util PRIVATE membrane_fs_util)
  target_include_directories(test_fs_util PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/unit)
  add_test(NAME test_fs_util COMMAND test_fs_util)

  add_executable(test_subprocess test_subprocess.cpp)
  set_target_properties(test_subprocess PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)
  target_link_libraries(test_subprocess PRIVATE membrane_subprocess)
  target_include_directories(test_subprocess PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/unit)
  add_test(NAME test_subprocess COMMAND test_subprocess)
endif()
