#ifndef __TEST_ASSERTIONS_H___
#define __TEST_ASSERTIONS_H___

// Simple assert macro for C
#define ASSERT_EQ(expr, expected, msg) \
    if ((expr) != (expected)) { \
        printf("[FAIL] %s (expected %d, got %d)\n", msg, (int)(expected), (int)(expr)); \
        return 1; \
    }
#define ASSERT_TRUE(expr, msg) \
    if (!(expr)) { \
        printf("[FAIL] %s\n", msg); \
        return 1; \
    }
#define ASSERT_STR_EQ(expr, expected, msg) \
    if (strcmp((expr), (expected)) != 0) { \
        printf("[FAIL] %s (expected %s, got %s)\n", msg, expected, expr); \
        return 1; \
    }

#endif // __TEST_ASSERTIONS_H___