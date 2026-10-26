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
#include <limits>
#include <stdexcept>

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

// Verifies matrix arithmetic operators and confirms non-mutating operators preserve their inputs.
static bool testMatrixArithmetic() {
	linalg::Matrix A(2, 2);
	A(0, 0) = 1;
	A(0, 1) = 2;
	A(1, 0) = 3;
	A(1, 1) = 4;

	linalg::Matrix B(2, 2);
	B(0, 0) = 5;
	B(0, 1) = 6;
	B(1, 0) = 7;
	B(1, 1) = 8;

	linalg::Matrix sum(2, 2);
	sum(0, 0) = 6;
	sum(0, 1) = 8;
	sum(1, 0) = 10;
	sum(1, 1) = 12;

	linalg::Matrix difference(2, 2);
	difference(0, 0) = -4;
	difference(0, 1) = -4;
	difference(1, 0) = -4;
	difference(1, 1) = -4;

	linalg::Matrix negativeA(2, 2);
	negativeA(0, 0) = -1;
	negativeA(0, 1) = -2;
	negativeA(1, 0) = -3;
	negativeA(1, 1) = -4;

	linalg::Matrix product(2, 2);
	product(0, 0) = 19;
	product(0, 1) = 22;
	product(1, 0) = 43;
	product(1, 1) = 50;

	if (A + B != sum || A - B != difference || -A != negativeA || A == B) {
		std::cerr << "FAILED: Non-mutating matrix arithmetic or exact inequality.\n";
		return false;
	}

	linalg::Matrix compound = A;
	compound += B;
	if (compound != sum) {
		std::cerr << "FAILED: Compound matrix addition.\n";
		return false;
	}

	compound -= B;
	if (compound != A) {
		std::cerr << "FAILED: Compound matrix subtraction.\n";
		return false;
	}

	compound *= B;
	if (compound != product || A(0, 0) != 1) {
		std::cerr << "FAILED: Compound matrix multiplication or input preservation.\n";
		return false;
	}

	std::cout << "PASSED: Matrix arithmetic operators.\n";
	return true;
}

// Checks approximate comparison for ordinary-sized values and incompatible dimensions.
static bool testIsApproxOrdinaryValues() {
	linalg::Matrix ordinaryA(1, 2);
	linalg::Matrix ordinaryB(1, 2);
	ordinaryA(0, 0) = 1.0;
	ordinaryA(0, 1) = -2.0;
	ordinaryB(0, 0) = 1.0005;
	ordinaryB(0, 1) = -2.001;
	if (!ordinaryA.isApprox(ordinaryB, 0.001, 0.0)
		|| ordinaryA.isApprox(ordinaryB, 0.0001, 0.0)
		|| ordinaryA.isApprox(ordinaryB)) {
		std::cerr << "FAILED: isApprox ordinary-value tolerance comparisons.\n";
		return false;
	}

	linalg::Matrix differentShape(2, 1);
	if (ordinaryA.isApprox(differentShape)) {
		std::cerr << "FAILED: isApprox accepted matrices with different dimensions.\n";
		return false;
	}

	std::cout << "PASSED: isApprox ordinary values and dimension mismatch.\n";
	return true;
}

// Checks that the absolute tolerance governs comparisons around zero.
static bool testIsApproxNearZero() {
	linalg::Matrix nearZeroA(1, 1);
	linalg::Matrix nearZeroB(1, 1);
	nearZeroA(0, 0) = 0.0;
	nearZeroB(0, 0) = 5e-13;
	if (!nearZeroA.isApprox(nearZeroB, 1e-12, 0.0)
		|| nearZeroA.isApprox(nearZeroB, 1e-13, 0.0)) {
		std::cerr << "FAILED: isApprox near-zero absolute tolerance comparisons.\n";
		return false;
	}

	std::cout << "PASSED: isApprox near-zero comparisons.\n";
	return true;
}

// Checks relative tolerance at a large scale, where absolute error alone is misleading.
static bool testIsApproxLargeValues() {
	linalg::Matrix largeA(1, 1);
	linalg::Matrix largeB(1, 1);
	largeA(0, 0) = 1e12;
	largeB(0, 0) = 1e12 + 500.0;
	if (!largeA.isApprox(largeB, 0.0, 1e-9)) {
		std::cerr << "FAILED: isApprox did not accept large values within relative tolerance.\n";
		return false;
	}

	largeB(0, 0) = 1e12 + 2000.0;
	if (largeA.isApprox(largeB, 0.0, 1e-9)) {
		std::cerr << "FAILED: isApprox accepted large values outside relative tolerance.\n";
		return false;
	}

	std::cout << "PASSED: isApprox large-value relative tolerance comparisons.\n";
	return true;
}

// Checks NaN rejection and matching/mismatched infinity handling.
static bool testIsApproxNonFiniteValues() {
	const double nan = std::numeric_limits<double>::quiet_NaN();
	const double infinity = std::numeric_limits<double>::infinity();
	linalg::Matrix nonFiniteA(1, 1);
	linalg::Matrix nonFiniteB(1, 1);
	nonFiniteA(0, 0) = nan;
	nonFiniteB(0, 0) = nan;
	if (nonFiniteA.isApprox(nonFiniteB)) {
		std::cerr << "FAILED: isApprox accepted NaN compared with NaN.\n";
		return false;
	}

	nonFiniteB(0, 0) = 1.0;
	if (nonFiniteA.isApprox(nonFiniteB)) {
		std::cerr << "FAILED: isApprox accepted NaN compared with a finite value.\n";
		return false;
	}

	nonFiniteA(0, 0) = infinity;
	nonFiniteB(0, 0) = infinity;
	if (!nonFiniteA.isApprox(nonFiniteB)) {
		std::cerr << "FAILED: isApprox rejected matching positive infinities.\n";
		return false;
	}

	nonFiniteB(0, 0) = -infinity;
	if (nonFiniteA.isApprox(nonFiniteB)) {
		std::cerr << "FAILED: isApprox accepted infinities with opposite signs.\n";
		return false;
	}

	nonFiniteB(0, 0) = 1.0;
	if (nonFiniteA.isApprox(nonFiniteB)) {
		std::cerr << "FAILED: isApprox accepted infinity compared with a finite value.\n";
		return false;
	}

	nonFiniteA(0, 0) = -infinity;
	nonFiniteB(0, 0) = -infinity;
	if (!nonFiniteA.isApprox(nonFiniteB, 1e-12, 1e-9)) {
		std::cerr << "FAILED: isApprox rejected matching negative infinities.\n";
		return false;
	}

	std::cout << "PASSED: isApprox NaN and infinity comparisons.\n";
	return true;
}

// Checks that invalid tolerance values are reported with invalid_argument.
static bool testIsApproxInvalidTolerances() {
	const double nan = std::numeric_limits<double>::quiet_NaN();
	const double infinity = std::numeric_limits<double>::infinity();
	linalg::Matrix finiteA(1, 1);
	linalg::Matrix finiteB(1, 1);
	finiteA(0, 0) = 1.0;
	finiteB(0, 0) = 1.0;
	if (!finiteA.isApprox(finiteB, 0.0, 0.0)) {
		std::cerr << "FAILED: isApprox rejected equal values with zero tolerances.\n";
		return false;
	}

	auto throwsInvalidTolerance = [&finiteA, &finiteB](double absTol, double relTol) {
		try {
			finiteA.isApprox(finiteB, absTol, relTol);
		}
		catch (const std::invalid_argument&) {
			return true;
		}
		return false;
	};

	if (!throwsInvalidTolerance(-1.0, 0.0)
		|| !throwsInvalidTolerance(0.0, -1.0)
		|| !throwsInvalidTolerance(nan, 0.0)
		|| !throwsInvalidTolerance(0.0, nan)
		|| !throwsInvalidTolerance(infinity, 0.0)
		|| !throwsInvalidTolerance(0.0, infinity)) {
		std::cerr << "FAILED: isApprox did not throw for an invalid tolerance.\n";
		return false;
	}

	std::cout << "PASSED: isApprox invalid tolerance handling.\n";
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

	allPassed = testMatrixArithmetic() && allPassed;

	allPassed = testIsApproxOrdinaryValues() && allPassed;

	allPassed = testIsApproxNearZero() && allPassed;

	allPassed = testIsApproxLargeValues() && allPassed;

	allPassed = testIsApproxNonFiniteValues() && allPassed;

	allPassed = testIsApproxInvalidTolerances() && allPassed;

	// Return the conventional process status: zero means success, nonzero means failure.
	return allPassed ? 0 : 1;
}