# Minimum required CMake version
cmake_minimum_required(VERSION 3.20)

# Enable Hot Reload for MSVC compilers if supported
if (POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)

    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT
        "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>"
    )
endif()

# Define the project
project(LinearAlgebra LANGUAGES CXX)

enable_testing()

# Use C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==========================================
# Linear Algebra Library
# ==========================================

add_library(linalg STATIC
    src/Matrix.cpp
    include/linalg/Matrix.hpp
   )

# Make the library's public headers available
target_include_directories(linalg PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ==========================================
# Example Executable
# ==========================================

add_executable(example
    examples/main.cpp
)

# Link the example program to our library
target_link_libraries(example PRIVATE linalg)

# ==========================================
# Test Executable
# ==========================================

add_executable(test_matrix
    tests/test_matrix.cpp
)

# Link the test program to our library
target_link_libraries(test_matrix PRIVATE linalg)

add_test(
    NAME MatrixTests
    COMMAND test_matrix
)