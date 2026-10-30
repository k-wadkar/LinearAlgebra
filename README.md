# LinearAlgebra

LinearAlgebra is a small C++20 library for working with  dense matrices of `double` values whose dimensions are chosen at construction time. It provides matrix arithmetic, row operations, Gaussian elimination, reduced row-echelon form, rank, and PLU factorisation with partial pivoting. The project also includes a runnable example, standalone tests, and multiplication benchmarks.

## Using the library

### Build

You need CMake 3.20 or newer and a C++20 compiler. During CMake configuration, the project downloads Eigen 3.4.0 and Google Benchmark 1.9.4 using CMake's `FetchContent`; configuring therefore requires network access.

For a general CMake generator and compiler already available in your environment:

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

On Windows, the supplied presets use Ninja and the MSVC `cl.exe` compiler. Run them from an environment where MSVC and Ninja are available:

```powershell
cmake --preset x64-debug
cmake --build out/build/x64-debug
ctest --test-dir out/build/x64-debug --output-on-failure
```

To build and test the release configuration instead, use `x64-release` and `out/build/x64-release`.

### A first example

Public types and functions are in the `linalg` namespace. Include the headers you use with paths beginning in `linalg/`:

```cpp
#include <iostream>
#include <linalg/Matrix.hpp>

int main() {
    linalg::Matrix A(2, 2);
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(1, 0) = 3.0;
    A(1, 1) = 4.0;

    linalg::Matrix B = linalg::transpose(A);
    linalg::Matrix C = A * B;

    C.toReducedEchelonForm();
    std::cout << C;
}
```

The constructor takes the number of rows and columns and initializes every entry to zero. Access or assign an entry with `matrix(row, column)`, using zero-based indices. The non-const form returns a writable reference; the const form allows reading. An invalid index throws `std::out_of_range`.

### Matrix API

| Operation | Description |
|---|---|
| `rows()`, `cols()` | Get the dimensions. |
| `matrix(i, j)` | Read or modify an element. |
| `A + B`, `A - B`, `-A` | Create a matrix sum, difference, or negation. Addition and subtraction require matching dimensions. |
| `A * B` | Matrix product; requires `A.cols() == B.rows()`. |
| `scalar * A`, `A * scalar` | Scale all entries. |
| `A += B`, `A -= B`, `A *= B` | Update a matrix in place. |
| `transpose(A)` | Return a new matrix with rows and columns exchanged. |
| `A.isApprox(B, absTol, relTol)` | Compare entries using `abs(a-b) <= absTol + relTol * max(abs(a), abs(b))`. Defaults are `1e-12` and `1e-9`. |
| `A.rank()` | Compute rank by elimination on a copy, leaving `A` unchanged. |
| `A.toEchelonForm(relTol)` | Convert `A` to echelon form in place. |
| `A.echelonForm(relTol)` | Return an echelon-form copy. |
| `A.toReducedEchelonForm(relTol)` | Convert `A` to reduced echelon form in place. |
| `A.reducedEchelonForm(relTol)` | Return a reduced-echelon-form copy. |
| `A.pluFactorisation()` | Return the permutation, lower, and upper factors. |
| `A.maxAbsoluteRealVal()` | Return the largest absolute entry; rejects NaN and infinity. |
| `std::cout << A` | Print the matrix in aligned scientific notation. |

Elimination methods accept a non-negative, finite relative tolerance. With the default `0.0`, only exact zeros are treated as zero. A positive tolerance sets the pivot cutoff to `relTol` times the largest absolute entry in the matrix. Mutating methods return a reference to the matrix, so row operations and in-place conversions can be chained.

`PermutationMatrix` represents a square permutation matrix without storing all its zero and one entries. Constructing it with a size creates the identity permutation. `swapRows(i, j)` changes the permutation, `operator()(i, j)` reads an entry, and `originalRow(i)` gives the source row now occupying destination row `i`. Multiplying it on the left by a matrix reorders that matrix's rows:

```cpp
linalg::PermutationMatrix P(3);
P.swapRows(0, 2);
linalg::Matrix reordered = P * A;
```

`Matrix::pluFactorisation()` returns a `linalg::PLUFactorisation` struct with fields `P`, `L`, and `U`. Its convention is `P * A = L * U`. `L` has ones on its diagonal; row swaps are recorded in `P` and reflected in the already-computed part of `L`. The method accepts rectangular and rank-deficient matrices as well as square ones.

Dimension mismatches and invalid arguments are reported with standard exceptions such as `std::invalid_argument`; invalid element or row indices use `std::out_of_range`. Callers can catch these through the standard exception types.

## Implementation and repository guide

### How the library is organized

C++ header files (`.hpp`) declare the types and functions that users can include. Source files (`.cpp`) contain compiled function definitions. The `linalg` CMake target builds these sources into a static library, and its public include directory lets applications write includes such as `#include <linalg/Matrix.hpp>`. The CMake targets for the example, tests, and benchmark link to that library.

The matrix stores its dimensions and its entries in a `std::vector<double>`. Entries are stored in row-major order: all columns of one row are contiguous in memory. An element at row `i`, column `j` is stored at offset `i * cols + j`. `std::size_t` is used for dimensions and indices because it is the standard unsigned type used for container sizes.

### Algorithms and data structures

**Matrix arithmetic.** Addition and subtraction visit corresponding entries. Matrix multiplication checks that the inner dimensions agree, initializes a zero result, and uses an `i-k-j` loop order. The innermost loop traverses columns of a row in row-major storage, which helps use contiguous memory. Transpose creates a new matrix and writes each entry at the exchanged row and column.

**Gaussian elimination and rank.** The elimination routine scans columns from left to right and advances the pivot row only when it finds a usable pivot. For each column it chooses the remaining row with the largest absolute entry (partial pivoting), swaps that row into place, and eliminates entries below the pivot. This handles rectangular matrices and skipped pivot columns. Row elimination uses `std::fma` for the row combination and explicitly writes the eliminated pivot-column entry as zero. The routine can also record row swaps and elimination multipliers for PLU factorisation. Rank is the number of pivot columns found while reducing a copy.

**Reduced row-echelon form.** The code first performs forward elimination, then visits the pivot columns in reverse order. It divides each pivot row so its pivot becomes one, and eliminates the entries above that pivot. Non-mutating echelon-form functions copy the input and apply the corresponding in-place operation to the copy.

**PLU factorisation.** Factorisation copies the input to `U`, starts `P` as the identity permutation, and initializes `L` with ones on its diagonal. The shared elimination routine updates `U`, records row swaps in `P`, and stores each elimination multiplier below the diagonal of `L`. When a pivot swap occurs, the previously stored entries of the affected rows in `L` are swapped too. This maintains the stated `P * A = L * U` convention, including for rectangular matrices.

**Permutation matrices.** A permutation matrix is represented by a vector of row indices rather than a dense matrix. Entry `(i, j)` is one exactly when the stored source-column index for row `i` equals `j`, and zero otherwise. Left multiplication uses those indices to copy rows directly, avoiding a dense matrix multiplication.

### Files

| Path | Purpose |
|---|---|
| `CMakeLists.txt` | Declares the C++20 project, downloads Eigen and Google Benchmark, and defines the library, example, test, and benchmark build targets. It also registers the tests with CTest. |
| `CMakePresets.json` | Provides Windows Ninja/MSVC configure presets for x64 and x86 Debug and Release builds. |
| `include/linalg/Matrix.hpp` | Public declaration of the `Matrix` class and the free arithmetic, output, and transpose functions. |
| `src/Matrix.cpp` | Implements matrix storage access, arithmetic, row operations, elimination, rank, approximate comparison, PLU construction, and stream output. |
| `include/linalg/PermutationMatrix.hpp` | Public declaration of the compact `PermutationMatrix` class, its row operations, multiplication, and output. |
| `src/PermutationMatrix.cpp` | Implements permutation-vector storage, bounds-checked access, row swapping, row-reordering multiplication, and output. |
| `include/linalg/PLUFactorisation.hpp` | Defines `PLUFactorisation`, the result struct holding `P`, `L`, and `U`. |
| `src/PLUFactorisation.cpp` | Present in the current CMake target but empty; the factorisation logic is implemented by `Matrix::pluFactorisation()` in `src/Matrix.cpp`. |
| `examples/main.cpp` | A small executable demonstrating permutation-matrix construction, row swaps, and printing. |
| `tests/test_matrix.cpp` | Standalone checks for matrix operations, elimination, rank, and approximate comparisons. |
| `tests/test_permutation_matrix.cpp` | Standalone checks for permutation construction, row swaps, indexing, multiplication, and output. |
| `tests/test_PLU_factorisation.cpp` | Standalone checks of PLU factors and the reconstruction relation across square and rectangular cases. |
| `benchmarks/benchmark_multiplication.cpp` | Google Benchmark comparisons of multiplication loop orders, this library's multiplication, and Eigen's multiplication. |
| `LICENSE.txt` | The project's MIT license text. |

### Tests and benchmarks

The tests are ordinary C++ executables rather than tests built with a separate unit-testing framework. CMake registers each executable with CTest. To run just one group after building:

```sh
ctest --test-dir build -R MatrixTests --output-on-failure
ctest --test-dir build -R PermutationMatrixTests --output-on-failure
ctest --test-dir build -R PLUFactorisationTests --output-on-failure
```

Build the `benchmark_multiplication` target and run the resulting executable to collect timing results. The benchmark source includes several multiplication loop-order variants; only selected variants are currently registered to run, alongside the Eigen and library comparisons. Benchmarks measure performance on the current machine and are not correctness tests.
