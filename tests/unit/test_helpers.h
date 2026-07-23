#ifndef MEMBRANE_TEST_HELPERS_H
#define MEMBRANE_TEST_HELPERS_H

#include <stdio.h>
#include <stdlib.h>

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, (msg)); \
            abort(); \
        } \
    } while (0)

#endif /* MEMBRANE_TEST_HELPERS_H */
