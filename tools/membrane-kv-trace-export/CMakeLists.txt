# Deliberately unconditional (no MEMBRANE_ENABLE_LLAMA gate): this tool
# only reads an already-captured membrane/kvdump.h file and writes a
# .memkv trace -- it never includes or links llama.cpp.
add_executable(membrane-kv-trace-export main.c)
target_link_libraries(membrane-kv-trace-export PRIVATE membrane_core
  membrane_workload_core)
