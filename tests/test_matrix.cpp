// For testing, use the x64 Developer Command Prompt with working directory LinearAlgebra

// For debug testing, run: 
// cmake --build out/build/x64-debug --target test_matrix
// ctest --test-dir out/build/x64-debug -V

// For release testing run:
// cmake --build out/build/x64-release --target test_matrix
// ctest --test-dir out/build/x64-release -V


#include <linalg/Matrix.hpp>
#include <iostream>
#include <iomanip>

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

	E(0, 0) = 4;
	E(0, 1) = 10;
	E(0, 2) = 6;

	E(1, 0) = 0;
	E(1, 1) = 3;
	E(1, 2) = 7;

	E(2, 0) = 0;
	E(2, 1) = 0;
	E(2, 2) = 4.0/3;

	A.toEchelonForm();

	if (A != E) {
		std::cerr << "FAILED: Standard Gaussian elimination test.\n";
		
		std::cerr << "\nExpected:\n\n" << E << "\n\n Actual:\n\n" << A << "\n";
		
		std::cerr << std::setprecision(17);

		std::cerr << "Expected: " << E(2, 2) << '\n';
		std::cerr << "Actual:   " << A(2, 2) << '\n';
		std::cerr << "Difference: " << A(2, 2) - E(2, 2) << '\n';

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

	E(0, 0) = 4;
	E(0, 1) = 6;
	E(0, 2) = 8;

	E(1, 0) = 0;
	E(1, 1) = 2;
	E(1, 2) = 4;

	E(2, 0) = 0;
	E(2, 1) = 0;
	E(2, 2) = 0;

	B.toEchelonForm();
	
	if (B != E) {
		std::cerr << "FAILED: Gaussian elimination test with row swaps.\n";
		
		std::cerr << "\nExpected:\n\n" << E << "\n\n Actual:\n\n" << B << "\n";
		
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
	E(0, 1) = 6;
	E(0, 2) = 12;
	E(0, 3) = 18;

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

		std::cerr << "\nExpected:\n\n" << E << "\n\n Actual:\n\n" << C << "\n";

		return false;
	}

	std::cout << "PASSED: Gaussian elimination on a rectangular matrix.\n";
	return true;
}

// Verifies that basic Gauss-Jordan reduction works
static bool testGaussJordan() {
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

	E(0, 0) = 1;
	E(0, 1) = 0;
	E(0, 2) = -1;

	E(1, 0) = 0;
	E(1, 1) = 1;
	E(1, 2) = 2;

	E(2, 0) = 0;
	E(2, 1) = 0;
	E(2, 2) = 0;

	B.toReducedEchelonForm();

	if (B != E) {
		std::cerr << "FAILED: Gauss-Jordan reduction.\n";
		return false;
	}

	std::cout << "PASSED: Gauss-Jordan reduction.\n";
	return true;

}

// Verifies that Gauss-Jordan reduction works on a rectangular matrix
static bool testGaussJordanRectangular() {
	linalg::Matrix B(3, 4);

	B(0, 0) = 0;
	B(0, 1) = 2;
	B(0, 2) = 4;
	B(0, 3) = 6;

	B(1, 0) = 0;
	B(1, 1) = 4;
	B(1, 2) = 10;
	B(1, 3) = 14;

	B(2, 0) = 0;
	B(2, 1) = 6;
	B(2, 2) = 12;
	B(2, 3) = 18;

	linalg::Matrix E(3, 4);

	E(0, 0) = 0;
	E(0, 1) = 1;
	E(0, 2) = 0;
	E(0, 3) = 1;

	E(1, 0) = 0;
	E(1, 1) = 0;
	E(1, 2) = 1;
	E(1, 3) = 1;

	E(2, 0) = 0;
	E(2, 1) = 0;
	E(2, 2) = 0;
	E(2, 3) = 0;

	B.toReducedEchelonForm();

	if (B != E) {
		std::cerr << "FAILED: Gauss-Jordan reduction on a rectangular matrix.\n";
		return false;
	}

	std::cout << "PASSED: Gauss-Jordan reduction on a rectangular matrix.\n";
	return true;

}

// Verifies that Gauss-Jordan reduction works on a matrix where there is a pivot in the final column
static bool testGaussJordanFinalColPivot() {
	linalg::Matrix B(2, 4);

	B(0, 0) = 0;
	B(0, 1) = 0;
	B(0, 2) = 2;
	B(0, 3) = 4;

	B(1, 0) = 0;
	B(1, 1) = 0;
	B(1, 2) = 0;
	B(1, 3) = 3;

	linalg::Matrix E(2, 4);

	E(0, 0) = 0;
	E(0, 1) = 0;
	E(0, 2) = 1;
	E(0, 3) = 0;

	E(1, 0) = 0;
	E(1, 1) = 0;
	E(1, 2) = 0;
	E(1, 3) = 1;

	B.toReducedEchelonForm();

	if (B != E) {
		std::cerr << "FAILED: Gauss-Jordan reduction on a rectangular matrix with a pivot in the final column.\n";
		return false;
	}

	std::cout << "PASSED: Gauss-Jordan reduction on a rectangular matrix with a pivot in the final column.\n";
	return true;

}

// Verifies that rank is calculated correctly
static bool testRank() {
	linalg::Matrix A(2, 2);

	A(0, 0) = 1;
	A(0, 1) = 2;
	A(1, 0) = 2;
	A(1, 1) = 4;

	if (A.rank() != 1) {
		std::cerr << "FAILED: Calculate rank of 2x2 matrix.\n";
		return false;
	}

	std::cout << "PASSED: Calculate rank of 2x2 matrix. \n";
	return true;

}

// Verifies that rank is calculated correctly on a full-rank matrix
static bool testFullRank() {
	linalg::Matrix A(2, 2);

	A(0, 0) = 1;
	A(0, 1) = 2;
	A(1, 0) = 3;
	A(1, 1) = 4;

	if (A.rank() != 2) {
		std::cerr << "FAILED: Calculate rank of full-rank matrix.\n";
		return false;
	}

	std::cout << "PASSED: Calculate rank of full-rank matrix. \n";
	return true;

}

// Verifies that non-mutating Gauss-Jordan reduction works
static bool testNonMutieGaussJordan() {
	linalg::Matrix A(2, 2);

	A(0, 0) = 1;
	A(0, 1) = 2;
	A(1, 0) = 2;
	A(1, 1) = 4;

	linalg::Matrix copy = A;

	linalg::Matrix reduced = A.reducedEchelonForm();

	linalg::Matrix E(2, 2);

	E(0, 0) = 1;
	E(0, 1) = 2;
	E(1, 0) = 0;
	E(1, 1) = 0;

	if (reduced != E) {
		std::cerr << "FAILED: Non-mutating Gauss-Jordan reduction.";
		return false;
	}
	else if (A != copy) {
		std::cerr << "FAILED: Original matrix was modified.\n";
		return false;
	}

	std::cout << "PASSED: Non-mutating Gauss-Jordan reduction.";
	return true;
}



int main() {
	bool allPassed = true;
	
	// Keep running every test even if an earlier test fails.
	
	// allPassed = testGaussianElimination() && allPassed;

	allPassed = testGaussianEliminationRowSwap() && allPassed;

	allPassed = testGaussianEliminationRectangular() && allPassed;

	allPassed = testGaussJordan() && allPassed;

	allPassed = testGaussJordanRectangular() && allPassed;

	allPassed = testGaussJordanFinalColPivot() && allPassed;

	allPassed = testRank() && allPassed;
	
	allPassed = testFullRank() && allPassed;

	allPassed = testNonMutieGaussJordan() && allPassed;

	// Return the conventional process status: zero means success, nonzero means failure.
	return allPassed ? 0 : 1;
}