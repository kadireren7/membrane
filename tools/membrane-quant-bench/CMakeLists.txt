add_executable(membrane-quant-bench main.c)
target_link_libraries(membrane-quant-bench PRIVATE membrane_core)

# The allocation/free counters in main.c are implemented with --wrap,
# which is a LINKER capability (GNU ld, gold, lld), not a compiler one.
# The previous CMAKE_C_COMPILER_ID guard asked the wrong question: on
# macOS the compiler id is "AppleClang", which still matches a
# "GNU|Clang" regex, so these flags were applied to Apple's own ld --
# which has no --wrap at all ("ld: unknown options: --wrap=malloc") and
# then fails the link outright ("Undefined symbols for architecture
# arm64: ___real_malloc ... ___real_realloc"). Probing the real linker
# answers the actual question and keeps the counters wherever they
# genuinely work, on every current and future toolchain, instead of
# hardcoding a compiler-name list.
include(CheckCSourceCompiles)
set(CMAKE_REQUIRED_LINK_OPTIONS -Wl,--wrap=malloc)
check_c_source_compiles("
#include <stdlib.h>
extern void *__real_malloc(size_t n);
void *__wrap_malloc(size_t n) { return (__real_malloc(n)); }
int main(void) { return (__wrap_malloc(1) == NULL); }
" MEMBRANE_HAVE_LD_WRAP)
unset(CMAKE_REQUIRED_LINK_OPTIONS)

if(MEMBRANE_HAVE_LD_WRAP)
  target_compile_definitions(membrane-quant-bench PRIVATE QBENCH_HAVE_WRAP=1)
  target_link_options(membrane-quant-bench PRIVATE
    -Wl,--wrap=malloc
    -Wl,--wrap=free
    -Wl,--wrap=calloc
    -Wl,--wrap=realloc)
endif()
