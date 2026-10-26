// For testing, use the x64 Developer Command Prompt with working directory LinearAlgebra

// For debug testing, run: 
// cmake --build out/build/x64-debug --target test_matrix
// ctest --test-dir out/build/x64-debug -V

// For release testing run:
// cmake --build out/build/x64-release --target test_matrix
// ctest --test-dir out/build/x64-release -V


#include <linalg/Matrix.hpp>
#include <iostream>

// static gives this test helper internal linkage, so it is only visible in this source file.
// It verifies elimination when each pivot is already in place.
static bool testGaussianElimination() {
	linalg::Matrix A(3, 3);

	A(0, 0) = 2;
	A(0, 1) = 4;
	A(0, 2) = 2;

	A(1, 0) = 4;
	A(1, 1) = 10;
	A(1, 2) = 6;

	A(2, 0) = 2;
	A(2, 1) = 8;
	A(2, 2) = 10;

	linalg::Matrix E(3, 3);

	E(0, 0) = 2;
	E(0, 1) = 4;
	E(0, 2) = 2;

	E(1, 0) = 0;
	E(1, 1) = 2;
	E(1, 2) = 2;

	E(2, 0) = 0;
	E(2, 1) = 0;
	E(2, 2) = 4;

	A.toEchelonForm();

	if (A != E) {
		std::cerr << "FAILED: Standard Gaussian elimination test.\n";
		return false;
	}

	std::cout << "PASSED: Standard Gaussian elimination test.\n";
	return true;
}

// Verifies that elimination swaps in a usable pivot row when the current one is zero.
static bool testGaussianEliminationRowSwap(){
	linalg::Matrix B(3, 3);

	B(0, 0) = 0;
	B(0, 1) = 2;
	B(0, 2) = 4;

	B(1, 0) = 2;
	B(1, 1) = 4;
	B(1, 2) = 6;

	B(2, 0) = 4;
	B(2, 1) = 6;
	B(2, 2) = 8;

	linalg::Matrix E(3, 3);

	E(0, 0) = 2;
	E(0, 1) = 4;
	E(0, 2) = 6;

	E(1, 0) = 0;
	E(1, 1) = 2;
	E(1, 2) = 4;

	E(2, 0) = 0;
	E(2, 1) = 0;
	E(2, 2) = 0;

	B.toEchelonForm();
	
	if (B != E) {
		std::cerr << "FAILED: Gaussian elimination test with row swaps.\n";
		return false;
	}

	std::cout << "PASSED: Gaussian elimination test with row swaps.\n";
	return true;

}

// Verifies pivot-column scanning and elimination for a non-square matrix.
static bool testGaussianEliminationRectangular() {
	linalg::Matrix C(3, 4);

	C(0, 0) = 0;
	C(0, 1) = 2;
	C(0, 2) = 4;
	C(0, 3) = 6;

	C(1, 0) = 0;
	C(1, 1) = 4;
	C(1, 2) = 10;
	C(1, 3) = 14;

	C(2, 0) = 0;
	C(2, 1) = 6;
	C(2, 2) = 12;
	C(2, 3) = 18;

	linalg::Matrix E(3, 4);

	E(0, 0) = 0;
	E(0, 1) = 2;
	E(0, 2) = 4;
	E(0, 3) = 6;

	E(1, 0) = 0;
	E(1, 1) = 0;
	E(1, 2) = 2;
	E(1, 3) = 2;

	E(2, 0) = 0;
	E(2, 1) = 0;
	E(2, 2) = 0;
	E(2, 3) = 0;

	C.toEchelonForm();

	if (C != E) {
		std::cerr << "FAILED: Gaussian elimination on a rectangular matrix.\n";
		return false;
	}

	std::cout << "PASSED: Gaussian elimination on a rectangular matrix.\n";
	return true;
}


int main() {
	bool allPassed = true;
	
	// Keep running every test even if an earlier test fails.
	allPassed = testGaussianElimination() && allPassed;

	allPassed = testGaussianEliminationRowSwap() && allPassed;

	allPassed = testGaussianEliminationRectangular() && allPassed;

	// Return the conventional process status: zero means success, nonzero means failure.
	return allPassed ? 0 : 1;
}