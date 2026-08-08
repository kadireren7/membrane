add_library(membrane_workload_core STATIC
  workload_gen.c
  precision_policy.c
)
target_include_directories(membrane_workload_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(membrane_workload_core PUBLIC membrane_core)
