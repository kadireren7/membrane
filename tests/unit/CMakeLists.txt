foreach(t test_block test_codec_raw test_codec_rle test_store test_kvdump)
  add_executable(${t} ${t}.c)
  target_link_libraries(${t} PRIVATE membrane_core)
  add_test(NAME ${t} COMMAND ${t})
endforeach()

# test_store, test_store_backend and test_store_concurrent are white-box:
# they reach into store_internal.h to drive pin/eviction/state paths.
target_include_directories(test_store
  PRIVATE ${CMAKE_SOURCE_DIR}/src/store)

foreach(t test_store_backend test_store_concurrent)
  add_executable(${t} ${t}.c)
  target_link_libraries(${t} PRIVATE membrane_core)
  target_include_directories(${t} PRIVATE ${CMAKE_SOURCE_DIR}/src/store)
  add_test(NAME ${t} COMMAND ${t})
endforeach()

# test_block deliberately provokes a giant malloc to exercise the
# ALLOC_FAILED path; the sanitizer allocators must return NULL there
# instead of aborting (ASan and TSan use different option env vars).
set_tests_properties(test_block PROPERTIES ENVIRONMENT
  "ASAN_OPTIONS=allocator_may_return_null=1;TSAN_OPTIONS=allocator_may_return_null=1")
