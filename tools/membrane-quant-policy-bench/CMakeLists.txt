add_library(membrane_bench_core STATIC
  bench_core.c
  bench_io.c
  trace_set.c
)
target_include_directories(membrane_bench_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(membrane_bench_core PUBLIC membrane_core
  membrane_workload_core)

add_executable(membrane-quant-policy-bench main.c)
target_link_libraries(membrane-quant-policy-bench PRIVATE membrane_bench_core)
