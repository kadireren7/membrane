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
  membrane_adaptive_kv_policy membrane_gpu_device llama)
target_include_directories(membrane-run PRIVATE
  ${CMAKE_SOURCE_DIR}/tools/membrane-llama-runtime
  ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include)

# Low-risk install target (Section 17): membrane-run only. Callers
# still build with -DMEMBRANE_ENABLE_LLAMA=ON; this target doesn't
# change that requirement, it just makes the product binary
# installable once built: cmake --install build-llama.
#
# llama.cpp/ggml default to BUILD_SHARED_LIBS=ON even without an
# explicit -DBUILD_SHARED_LIBS flag (confirmed on a fresh configure --
# not this repo's own CMakeLists.txt setting it), so membrane-run is a
# dynamically-linked binary by default, not just in some unusual
# configuration. An install rule that only copies the executable would
# produce a binary that fails at startup with "cannot open shared
# object file" the moment it's run outside the build tree -- exactly
# the "install/build flow is broken" stop condition this phase is
# required to avoid. RPATH $ORIGIN/../lib makes the installed binary
# find its own installed libraries without requiring LD_LIBRARY_PATH.
set_target_properties(membrane-run PROPERTIES
  INSTALL_RPATH "$ORIGIN/../lib"
  BUILD_WITH_INSTALL_RPATH FALSE
  SKIP_BUILD_RPATH FALSE)
install(TARGETS membrane-run RUNTIME DESTINATION bin)
if(BUILD_SHARED_LIBS)
  # ggml-blas is optional (only built when GGML_BLAS=ON, not this
  # project's default) -- the TARGET guard below skips it cleanly when
  # absent. Included here so that IF a caller does enable GGML_BLAS,
  # its shared library still gets the same RPATH/install treatment as
  # the rest -- ggml links against it, and RUNPATH is not transitively
  # inherited (see the comment below), so an omitted ggml-blas would
  # fail to load post-install exactly like the bug this whole block
  # exists to fix.
  foreach(_membrane_run_dep IN ITEMS llama ggml ggml-base ggml-cpu
      ggml-blas membrane_core)
    if(TARGET ${_membrane_run_dep})
      # libllama.so/libggml*.so ship with no RPATH of their own (not
      # this project's target -- llama.cpp's own CMake sets none), so
      # libllama.so can locate its sibling libggml*.so only via
      # membrane-run's RUNPATH -- which the dynamic linker does NOT
      # apply transitively to a dependency-of-a-dependency (RUNPATH,
      # unlike the older RPATH mechanism, is not inherited down the
      # dependency chain). Each installed library needs its own
      # $ORIGIN so it can find its co-installed siblings directly.
      set_target_properties(${_membrane_run_dep} PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH FALSE
        SKIP_BUILD_RPATH FALSE)
      install(TARGETS ${_membrane_run_dep}
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION lib)
    endif()
  endforeach()
endif()
