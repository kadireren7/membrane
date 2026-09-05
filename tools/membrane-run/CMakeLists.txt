# Phase 9B.1: the one place membrane-run touches ggml_backend_dev_*()/
# gguf_*() directly (gpu_device.h's own header comment). Needs
# llama.h/ggml-backend.h/gguf.h, so -- unlike membrane_gpu_policy's
# pure arithmetic, registered unconditionally at the top level -- this
# library only exists inside the MEMBRANE_ENABLE_LLAMA block.
add_library(membrane_gpu_device STATIC gpu_device.cpp)
set_target_properties(membrane_gpu_device PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)
target_link_libraries(membrane_gpu_device PRIVATE llama)
target_include_directories(membrane_gpu_device PUBLIC
  ${CMAKE_SOURCE_DIR}/tools/membrane-run)
target_include_directories(membrane_gpu_device PRIVATE
  ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include)

# Mega Phase A, PR A1: the reusable runtime-session core (runtime_session.h's
# own top comment has the full contract) -- model lifecycle, planner
# orchestration (membrane_resolve_gpu_config()/membrane_resolve_cpu_
# adaptive_kv()/membrane_resolve_kv_placement()/membrane_resolve_ctx_auto(),
# extracted from main.cpp), and generation (membrane_session_generate(),
# wrapping the existing bounded apply-time fallback controller). Needs
# llama/ggml types (model/context lifetime, tokenize), so -- like
# membrane_gpu_device/membrane_decode_loop -- only built here, inside the
# MEMBRANE_ENABLE_LLAMA block. membrane-run (below) links this instead of
# implementing its own copy of any of this; a future server links it too.
add_library(membrane_runtime_session STATIC runtime_session.cpp)
set_target_properties(membrane_runtime_session PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)
target_link_libraries(membrane_runtime_session PUBLIC membrane_product_cli
  membrane_compat_check membrane_decode_loop membrane_gpu_policy
  membrane_adaptive_kv_policy membrane_kv_residency_policy
  membrane_joint_planner membrane_auto_fallback membrane_gpu_device
  membrane_context_recommender membrane_host_memory_guard
  membrane_context_auto_cli membrane_kv_store_telemetry llama)
target_include_directories(membrane_runtime_session PUBLIC
  ${CMAKE_SOURCE_DIR}/tools/membrane-run)
target_include_directories(membrane_runtime_session PRIVATE
  ${CMAKE_SOURCE_DIR}/tools/membrane-llama-runtime
  ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include)

if(MEMBRANE_BUILD_TESTS)
  # Mega Phase A, PR A1, Section 7: CI-safe unit tests for the
  # runtime-session core -- no real GGUF model file required (see
  # test_runtime_session.cpp's own top comment). The real small-model
  # smoke ("3 sequential requests from one loaded model") is the separate
  # runtime-session-smoke dev harness below, not this ctest target.
  add_executable(test_runtime_session test_runtime_session.cpp)
  set_target_properties(test_runtime_session PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)
  target_link_libraries(test_runtime_session PRIVATE membrane_runtime_session
    llama)
  target_include_directories(test_runtime_session PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/unit
    ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include)
  add_test(NAME test_runtime_session COMMAND test_runtime_session)
endif()

# Mega Phase A, PR A1, Section 7: a dev-only real-evidence harness (same
# precedent as context-recommender-dryrun below) proving "load model once,
# generate multiple requests, keep model alive, free cleanly" against a
# REAL local GGUF fixture -- not ctest-registered (needs a real model file
# CI does not have), not installed. Run manually; its output is the real
# evidence this phase's PR description/docs cite.
add_executable(runtime-session-smoke runtime_session_smoke.cpp)
set_target_properties(runtime-session-smoke PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)
target_link_libraries(runtime-session-smoke PRIVATE membrane_runtime_session
  llama)
target_include_directories(runtime-session-smoke PRIVATE
  ${CMAKE_SOURCE_DIR}/tools/membrane-llama-runtime
  ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include)

# Phase 33, Section 30: a dev-only real-evidence harness for the new
# context recommendation core -- NOT installed, NOT a product CLI
# surface (no --ctx auto ships this phase). Only built inside this
# MEMBRANE_ENABLE_LLAMA block because it needs real ggml_row_size()/
# GGUF device facts, exactly like membrane-run itself below.
add_executable(context-recommender-dryrun context_recommender_dryrun.cpp)
set_target_properties(context-recommender-dryrun PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)
target_link_libraries(context-recommender-dryrun PRIVATE
  membrane_context_recommender membrane_gpu_device
  membrane_kv_store_telemetry llama)
target_include_directories(context-recommender-dryrun PRIVATE
  ${CMAKE_SOURCE_DIR}/tools/membrane-llama-runtime
  ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include)

# Only evaluated when MEMBRANE_ENABLE_LLAMA=ON (see the top-level
# CMakeLists.txt's if(MEMBRANE_ENABLE_LLAMA) block). Product Phase 8:
# membrane-run, the user-facing entry point -- see main.cpp's header
# comment for the normal-mode/compare-mode split.
add_executable(membrane-run main.cpp)
set_target_properties(membrane-run PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)
target_link_libraries(membrane-run PRIVATE membrane_runtime_session
  membrane_product_cli membrane_compat_check membrane_decode_loop
  membrane_gpu_policy membrane_adaptive_kv_policy membrane_kv_residency_policy
  membrane_joint_planner membrane_auto_fallback membrane_gpu_device
  membrane_context_recommender membrane_host_memory_guard
  membrane_context_auto_cli membrane_kv_store_telemetry llama)
target_include_directories(membrane-run PRIVATE
  ${CMAKE_SOURCE_DIR}/tools/membrane-llama-runtime
  ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include)

# Install target: membrane-run only. Callers still build with
# -DMEMBRANE_ENABLE_LLAMA=ON; this target doesn't change that
# requirement, it just makes the product binary installable once
# built: cmake --install build-llama.
#
# llama.cpp/ggml default to BUILD_SHARED_LIBS=ON even without an
# explicit -DBUILD_SHARED_LIBS flag (confirmed on a fresh configure --
# not this repo's own CMakeLists.txt setting it), so membrane-run is a
# dynamically-linked binary by default, not just in some unusual
# configuration. An install rule that only copies the executable would
# produce a binary that fails at startup with "cannot open shared
# object file" the moment it's run outside the build tree -- exactly
# the "install/build flow is broken" stop condition this phase is
# required to avoid. RPATH $ORIGIN/<relative path to libdir> makes the
# installed binary find its own installed libraries without requiring
# LD_LIBRARY_PATH -- computed via file(RELATIVE_PATH ...) rather than
# a hardcoded "../lib" so it stays correct if CMAKE_INSTALL_LIBDIR ever
# resolves to something other than "lib" (e.g. "lib64" on some distros
# under a non-default GNUInstallDirs configuration).
file(RELATIVE_PATH MEMBRANE_RUN_RPATH_TO_LIBDIR
  "${CMAKE_INSTALL_FULL_BINDIR}" "${CMAKE_INSTALL_FULL_LIBDIR}")
set_target_properties(membrane-run PROPERTIES
  INSTALL_RPATH "$ORIGIN/${MEMBRANE_RUN_RPATH_TO_LIBDIR}"
  BUILD_WITH_INSTALL_RPATH FALSE
  SKIP_BUILD_RPATH FALSE)
install(TARGETS membrane-run RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES ${CMAKE_SOURCE_DIR}/LICENSE
  DESTINATION ${CMAKE_INSTALL_DATADIR}/membrane
  RENAME LICENSE.txt)
if(BUILD_SHARED_LIBS)
  # ggml-blas is optional (only built when GGML_BLAS=ON, not this
  # project's default) -- the TARGET guard below skips it cleanly when
  # absent. ggml-vulkan only exists on a -DGGML_VULKAN=ON configure;
  # the same guard skips it cleanly on a CPU-only build. Both are
  # included here so that whichever backends a given configure
  # actually produces, every one of them gets the same RPATH/install
  # treatment as the rest -- an omitted backend library fails to load
  # post-install exactly like the bug this whole block exists to fix
  # (confirmed directly: a Vulkan build's installed binary failed with
  # "cannot open shared object file: libggml-vulkan.so.0" before
  # ggml-vulkan was added to this list -- see docs/install.md).
  # Mega Phase D, PR D3: ggml-cuda joins this list the same way ggml-
  # vulkan did -- only exists on a -DGGML_CUDA=ON configure, the
  # existing TARGET guard below skips it cleanly otherwise.
  foreach(_membrane_run_dep IN ITEMS llama ggml ggml-base ggml-cpu
      ggml-blas ggml-vulkan ggml-cuda membrane_core)
    if(TARGET ${_membrane_run_dep})
      # membrane_core's own SHARED/STATIC-ness follows BUILD_SHARED_LIBS
      # at the point its add_library() call is evaluated, at the top of
      # the root CMakeLists.txt -- BEFORE this MEMBRANE_ENABLE_LLAMA
      # block (and thus before llama.cpp's own CMake sets
      # BUILD_SHARED_LIBS=ON) ever runs. On a from-scratch configure it
      # is therefore STATIC even though this whole block only runs when
      # BUILD_SHARED_LIBS is ON by the time *this* file is processed --
      # confirmed directly: a fresh-clone build installed a stray,
      # unnecessary libmembrane_core.a here with no runtime purpose
      # (statically linked straight into membrane-run) before this
      # TYPE check was added. Skip anything that isn't actually a
      # shared library rather than assuming every name in the list
      # above is one.
      get_target_property(_membrane_run_dep_type ${_membrane_run_dep}
        TYPE)
      if(NOT _membrane_run_dep_type STREQUAL "SHARED_LIBRARY")
        continue()
      endif()
      # libllama.so/libggml*.so ship with no RPATH of their own (not
      # this project's target -- llama.cpp's own CMake sets none), so
      # libllama.so can locate its sibling libggml*.so only via
      # membrane-run's RUNPATH -- which the dynamic linker does NOT
      # apply transitively to a dependency-of-a-dependency (RUNPATH,
      # unlike the older RPATH mechanism, is not inherited down the
      # dependency chain). Each installed library needs its own
      # $ORIGIN so it can find its co-installed siblings directly.
      # llama/ggml declare PUBLIC_HEADER files (their own development
      # headers). We deliberately don't ship a development SDK here
      # (Section 32: membrane-run is the product, not a library other
      # projects link against) -- clearing the property, rather than
      # passing PUBLIC_HEADER DESTINATION, avoids both installing
      # those headers and CMake's "has PUBLIC_HEADER files but no
      # PUBLIC_HEADER DESTINATION" dev-warning on every configure.
      set_target_properties(${_membrane_run_dep} PROPERTIES
        PUBLIC_HEADER ""
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH FALSE
        SKIP_BUILD_RPATH FALSE)
      install(TARGETS ${_membrane_run_dep}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR})
    endif()
  endforeach()
endif()
