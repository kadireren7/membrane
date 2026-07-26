add_executable(membrane-cxl-sim
  main.cpp
  sim_engine.cpp
  workload.cpp
)
set_target_properties(membrane-cxl-sim PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)
target_link_libraries(membrane-cxl-sim PRIVATE membrane_core)
