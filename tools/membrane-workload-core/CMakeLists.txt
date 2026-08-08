add_library(membrane_workload_core STATIC
  workload_gen.c
  precision_policy.c
  trace_format.c
)
target_include_directories(membrane_workload_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR})
# trace_format.c reuses the CRC32 table already defined for
# membrane_block_checksum (src/block/crc32_table.h) so its streaming,
# incremental checksum is guaranteed bit-identical to that function's
# whole-buffer one, rather than duplicating the table -- the same
# pattern test_store.c uses to reach store_internal.h.
target_include_directories(membrane_workload_core PRIVATE
  ${CMAKE_SOURCE_DIR}/src/block)
target_link_libraries(membrane_workload_core PUBLIC membrane_core)
