add_executable(membrane-quant-bench main.c)
target_link_libraries(membrane-quant-bench PRIVATE membrane_core)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(membrane-quant-bench PRIVATE
    -Wl,--wrap=malloc
    -Wl,--wrap=free
    -Wl,--wrap=calloc
    -Wl,--wrap=realloc)
endif()
