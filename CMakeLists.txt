# Require CMake 3.20 or newer for the commands and policies used below.
cmake_minimum_required(VERSION 3.20)

# Use the modern MSVC debug-information policy when this CMake version knows it.
# The generator expression selects Edit and Continue for Debug/RelWithDebInfo.
if (POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)

    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT
        "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>"
    )
endif()

# Declare the project and enable C++ as its implementation language.
project(LinearAlgebra LANGUAGES CXX)

include(FetchContent)

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)

set(BUILD_TESTING OFF CACHE BOOL "Disable dependency tests" FORCE)

FetchContent_MakeAvailable(eigen)

# ==========================================
# Google Benchmark
# ==========================================

FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.4
)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable Google Benchmark tests" FORCE)

FetchContent_MakeAvailable(googlebenchmark)


# Register tests with CTest so they can be run after configuration and build.
enable_testing()

# Require C++20 for the library and its consumers; do not silently fall back.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==========================================
# Linear Algebra Library
# ==========================================

add_library(linalg STATIC
    src/Matrix.cpp
    include/linalg/Matrix.hpp
    src/PermutationMatrix.cpp
    include/linalg/PermutationMatrix.hpp
    include/linalg/PLUFactorisation.hpp
    src/PLUFactorisation.cpp
)

# Build the reusable library as a static archive. Its header is listed for IDEs
# and project generators; Matrix.cpp supplies the compiled implementation.

# PUBLIC makes this include path available both while building linalg and to
# targets that link it, so consumers can include <linalg/Matrix.hpp>.
target_include_directories(linalg PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ==========================================
# Example Executable
# ==========================================

add_executable(example
    examples/main.cpp
)

# Link the example against linalg; PRIVATE means this dependency is for this
# executable rather than an interface that downstream targets inherit.
target_link_libraries(example PRIVATE linalg)

# ==========================================
# Matrix Test Executable
# ==========================================

add_executable(test_matrix
    tests/test_matrix.cpp
)

# The test executable uses the same library implementation as the example.
target_link_libraries(test_matrix PRIVATE linalg)

# Tell CTest how to run the built test executable.
add_test(
    NAME MatrixTests
    COMMAND test_matrix
)

# ==========================================
# Permutation Matrix Test Executable
# ==========================================

add_executable(test_permutation_matrix
    tests/test_permutation_matrix.cpp
)

# The test executable uses the same library implementation as the example.
target_link_libraries(test_permutation_matrix PRIVATE linalg)

# Tell CTest how to run the built test executable.
add_test(
    NAME PermutationMatrixTests
    COMMAND test_permutation_matrix
)

# ==========================================
# PLU Factorisation Test Executable
# ==========================================

add_executable(test_PLU_factorisation
    tests/test_PLU_factorisation.cpp
)

# The test executable uses the same library implementation as the example.
target_link_libraries(test_PLU_factorisation PRIVATE linalg)

# Tell CTest how to run the PLU factorisation tests.
add_test(
    NAME PLUFactorisationTests
    COMMAND test_PLU_factorisation
)


# ==========================================
# Benchmarks
# ==========================================

add_executable(benchmark_multiplication
    benchmarks/benchmark_multiplication.cpp
)

target_link_libraries(benchmark_multiplication PRIVATE
    linalg
    Eigen3::Eigen
)

target_link_libraries(benchmark_multiplication PRIVATE
    linalg
    Eigen3::Eigen
    benchmark::benchmark
)

set_property(TARGET linalg PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)

set_property(TARGET benchmark_multiplication PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)