foreach(t test_block test_codec_raw test_codec_rle)
  add_executable(${t} ${t}.c)
  target_link_libraries(${t} PRIVATE membrane_core)
  add_test(NAME ${t} COMMAND ${t})
endforeach()

# test_block deliberately provokes a giant malloc to exercise the
# ALLOC_FAILED path; ASan must return NULL there instead of aborting.
set_tests_properties(test_block PROPERTIES
  ENVIRONMENT "ASAN_OPTIONS=allocator_may_return_null=1")
