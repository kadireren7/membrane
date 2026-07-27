add_library(membrane_exactsim_core STATIC
  calibrate.cpp
  exact_engine.cpp
)
set_target_properties(membrane_exactsim_core PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)
target_include_directories(membrane_exactsim_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(membrane_exactsim_core PUBLIC
  membrane_wssim_core
  membrane_cxlsim_core)

add_executable(membrane-kv-exact-sim main.cpp)
set_target_properties(membrane-kv-exact-sim PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)
target_link_libraries(membrane-kv-exact-sim PRIVATE membrane_exactsim_core)
