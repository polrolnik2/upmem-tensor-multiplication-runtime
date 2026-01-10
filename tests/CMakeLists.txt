include(FetchContent)

# Fetch Google Test
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# Find all C++ test files
file(GLOB TEST_SOURCES test_*.cpp)

# Create a test executable for each test file
foreach(test_src ${TEST_SOURCES})
  get_filename_component(test_name ${test_src} NAME_WE)
  add_executable(${test_name} ${test_src})
  target_link_libraries(${test_name} PRIVATE pim_matmul gtest gtest_main)
  add_dependencies(${test_name} dpu_kernel)
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
