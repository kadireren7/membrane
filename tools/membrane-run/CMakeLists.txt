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

# Only evaluated when MEMBRANE_ENABLE_LLAMA=ON (see the top-level
# CMakeLists.txt's if(MEMBRANE_ENABLE_LLAMA) block). Product Phase 8:
# membrane-run, the user-facing entry point -- see main.cpp's header
# comment for the normal-mode/compare-mode split.
add_executable(membrane-run main.cpp)
set_target_properties(membrane-run PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)
target_link_libraries(membrane-run PRIVATE membrane_product_cli
  membrane_compat_check membrane_decode_loop membrane_gpu_policy
  membrane_adaptive_kv_policy membrane_kv_residency_policy
  membrane_joint_planner membrane_auto_fallback membrane_gpu_device llama)
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
  foreach(_membrane_run_dep IN ITEMS llama ggml ggml-base ggml-cpu
      ggml-blas ggml-vulkan membrane_core)
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
