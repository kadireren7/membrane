add_library(membrane_demo_core STATIC
  demo_core.c
  demo_io.c
)
target_include_directories(membrane_demo_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(membrane_demo_core PUBLIC membrane_core
  membrane_workload_core)

add_executable(membrane-demo main.c)
target_link_libraries(membrane-demo PRIVATE membrane_demo_core)
