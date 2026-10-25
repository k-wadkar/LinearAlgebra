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
# Test Executable
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