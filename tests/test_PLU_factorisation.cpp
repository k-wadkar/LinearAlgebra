// For testing, use the x64 Developer Command Prompt with working directory LinearAlgebra

// For debug testing, run:
// cmake --build out/build/x64-debug --target test_PLU_factorisation
// ctest --test-dir out/build/x64-debug -R PLUFactorisationTests -V

// For release testing run:
// cmake --build out/build/x64-release --target test_PLU_factorisation
// ctest --test-dir out/build/x64-release -R PLUFactorisationTests -V


#include <linalg/Matrix.hpp>
#include <linalg/PLUFactorisation.hpp>
#include <iostream>

// Convert a permutation matrix to dense form so it can be compared with its expected factor.
static linalg::Matrix toDenseMatrix(const linalg::PermutationMatrix& P) {
	linalg::Matrix denseP(P.size(), P.size());

	for (std::size_t i = 0; i < P.size(); i++) {
		for (std::size_t j = 0; j < P.size(); j++) {
			denseP(i, j) = P(i, j);
		}
	}

	return denseP;
}

// Print the input and calculated factors so test output shows exactly what was factorised.
static void printFactorisation(
	const char* testName,
	const linalg::Matrix& A,
	const linalg::PLUFactorisation& factors
) {
	std::cout << "\n=== " << testName << " ===\n";
	std::cout << "Original matrix A:\n" << A << '\n';
	std::cout << "Permutation matrix P:\n" << factors.P << '\n';
	std::cout << "Lower factor L:\n" << factors.L << '\n';
	std::cout << "Upper factor U:\n" << factors.U << '\n';
}

// Compare the returned factors with independently calculated expected factors.
static bool checkExpectedFactors(
	const char* testName,
	const linalg::Matrix& A,
	const linalg::Matrix& originalCopy,
	const linalg::PLUFactorisation& actual,
	const linalg::Matrix& expectedP,
	const linalg::Matrix& expectedL,
	const linalg::Matrix& expectedU
) {
	const linalg::Matrix actualP = toDenseMatrix(actual.P);
	const bool permutationMatches = actualP.isApprox(expectedP);
	const bool lowerFactorMatches = actual.L.isApprox(expectedL);
	const bool upperFactorMatches = actual.U.isApprox(expectedU);
	const bool reconstructs = (actualP * A).isApprox(actual.L * actual.U);
	const bool inputUnchanged = A.isApprox(originalCopy);

	if (!permutationMatches) {
		std::cerr << "FAILED: " << testName << " produced an unexpected P.\n"
			<< "Expected P:\n" << expectedP << "\nActual P:\n" << actualP << '\n';
	}
	if (!lowerFactorMatches) {
		std::cerr << "FAILED: " << testName << " produced an unexpected L.\n"
			<< "Expected L:\n" << expectedL << "\nActual L:\n" << actual.L << '\n';
	}
	if (!upperFactorMatches) {
		std::cerr << "FAILED: " << testName << " produced an unexpected U.\n"
			<< "Expected U:\n" << expectedU << "\nActual U:\n" << actual.U << '\n';
	}
	if (!reconstructs) {
		std::cerr << "FAILED: " << testName << " does not reconstruct P*A ~= L*U.\n"
			<< "P * A:\n" << actualP * A << "\nL * U:\n" << actual.L * actual.U << '\n';
	}
	if (!inputUnchanged) {
		std::cerr << "FAILED: " << testName << " modified the input matrix.\n";
	}

	if (!permutationMatches || !lowerFactorMatches || !upperFactorMatches
		|| !reconstructs || !inputUnchanged) {
		return false;
	}

	std::cout << "PASSED: " << testName << " (P, L, and U match their expected factors).\n";
	return true;
}

// Run factorisation, print the computed result, and check its factors against independent expectations.
static bool testPLUFactorisation(
	const char* testName,
	const linalg::Matrix& A,
	const linalg::Matrix& expectedP,
	const linalg::Matrix& expectedL,
	const linalg::Matrix& expectedU
) {
	const linalg::Matrix originalCopy = A;

	try {
		const linalg::PLUFactorisation actual = A.pluFactorisation();
		printFactorisation(testName, A, actual);
		return checkExpectedFactors(
			testName, A, originalCopy, actual, expectedP, expectedL, expectedU
		);
	}
	catch (const std::exception& error) {
		std::cerr << "FAILED: " << testName << " threw unexpectedly: " << error.what() << '\n';
		return false;
	}
}

// Verifies factors for a square matrix that does not need row pivoting.
static bool testPLUFactorisationWithoutPivoting() {
	linalg::Matrix A(3, 3);
	A(0, 0) = 4.0;
	A(0, 1) = 1.0;
	A(0, 2) = 2.0;
	A(1, 0) = 1.0;
	A(1, 1) = 5.0;
	A(1, 2) = 1.0;
	A(2, 0) = 2.0;
	A(2, 1) = 1.0;
	A(2, 2) = 6.0;

	linalg::Matrix expectedP(3, 3);
	expectedP(0, 0) = 1.0;
	expectedP(1, 1) = 1.0;
	expectedP(2, 2) = 1.0;

	linalg::Matrix expectedL(3, 3);
	expectedL(0, 0) = 1.0;
	expectedL(1, 0) = 1.0 / 4.0;
	expectedL(1, 1) = 1.0;
	expectedL(2, 0) = 1.0 / 2.0;
	expectedL(2, 1) = 2.0 / 19.0;
	expectedL(2, 2) = 1.0;

	linalg::Matrix expectedU(3, 3);
	expectedU(0, 0) = 4.0;
	expectedU(0, 1) = 1.0;
	expectedU(0, 2) = 2.0;
	expectedU(1, 1) = 19.0 / 4.0;
	expectedU(1, 2) = 1.0 / 2.0;
	expectedU(2, 2) = 94.0 / 19.0;

	return testPLUFactorisation("PLU without pivoting", A, expectedP, expectedL, expectedU);
}

// Verifies that a required first-column row swap is reflected in all three factors.
static bool testPLUFactorisationWithFirstPivotSwap() {
	linalg::Matrix A(3, 3);
	A(0, 0) = 0.0;
	A(0, 1) = 2.0;
	A(0, 2) = 1.0;
	A(1, 0) = 1.0;
	A(1, 1) = 1.0;
	A(1, 2) = 0.0;
	A(2, 0) = 2.0;
	A(2, 1) = 0.0;
	A(2, 2) = 1.0;

	linalg::Matrix expectedP(3, 3);
	expectedP(0, 2) = 1.0;
	expectedP(1, 0) = 1.0;
	expectedP(2, 1) = 1.0;

	linalg::Matrix expectedL(3, 3);
	expectedL(0, 0) = 1.0;
	expectedL(1, 1) = 1.0;
	expectedL(2, 0) = 1.0 / 2.0;
	expectedL(2, 1) = 1.0 / 2.0;
	expectedL(2, 2) = 1.0;

	linalg::Matrix expectedU(3, 3);
	expectedU(0, 0) = 2.0;
	expectedU(0, 2) = 1.0;
	expectedU(1, 1) = 2.0;
	expectedU(1, 2) = 1.0;
	expectedU(2, 2) = -1.0;

	return testPLUFactorisation("PLU with a first-pivot row swap", A, expectedP, expectedL, expectedU);
}

// Verifies later pivoting and the corresponding exchange of previously calculated L entries.
static bool testPLUFactorisationWithMultiplePivotSwaps() {
	linalg::Matrix A(3, 3);
	A(0, 0) = 1.0;
	A(0, 1) = 1.0;
	A(1, 0) = 10.0;
	A(1, 1) = 1.0;
	A(1, 2) = 1.0;
	A(2, 0) = 1.0;
	A(2, 1) = 20.0;
	A(2, 2) = 1.0;

	linalg::Matrix expectedP(3, 3);
	expectedP(0, 1) = 1.0;
	expectedP(1, 2) = 1.0;
	expectedP(2, 0) = 1.0;

	linalg::Matrix expectedL(3, 3);
	expectedL(0, 0) = 1.0;
	expectedL(1, 0) = 1.0 / 10.0;
	expectedL(1, 1) = 1.0;
	expectedL(2, 0) = 1.0 / 10.0;
	expectedL(2, 1) = 9.0 / 199.0;
	expectedL(2, 2) = 1.0;

	linalg::Matrix expectedU(3, 3);
	expectedU(0, 0) = 10.0;
	expectedU(0, 1) = 1.0;
	expectedU(0, 2) = 1.0;
	expectedU(1, 1) = 199.0 / 10.0;
	expectedU(1, 2) = 9.0 / 10.0;
	expectedU(2, 2) = -28.0 / 199.0;

	return testPLUFactorisation("PLU with multiple pivot swaps", A, expectedP, expectedL, expectedU);
}

// Verifies the returned factors for a one-by-one nonsingular matrix.
static bool testPLUFactorisationOneByOne() {
	linalg::Matrix A(1, 1);
	A(0, 0) = -7.5;

	linalg::Matrix expectedP(1, 1);
	expectedP(0, 0) = 1.0;
	linalg::Matrix expectedL(1, 1);
	expectedL(0, 0) = 1.0;
	linalg::Matrix expectedU(1, 1);
	expectedU(0, 0) = -7.5;

	return testPLUFactorisation("One-by-one PLU", A, expectedP, expectedL, expectedU);
}

// Verifies that the empty matrix produces empty factors with matching dimensions.
static bool testPLUFactorisationEmptyMatrix() {
	linalg::Matrix A(0, 0);
	linalg::Matrix expectedP(0, 0);
	linalg::Matrix expectedL(0, 0);
	linalg::Matrix expectedU(0, 0);

	return testPLUFactorisation("Empty-matrix PLU", A, expectedP, expectedL, expectedU);
}

// Verifies the rectangular factor shapes and values for a full-row-rank wide matrix.
static bool testPLUFactorisationWideMatrix() {
	linalg::Matrix A(2, 3);
	A(0, 0) = 1.0;
	A(0, 1) = 2.0;
	A(0, 2) = 3.0;
	A(1, 0) = 4.0;
	A(1, 1) = 5.0;
	A(1, 2) = 6.0;

	linalg::Matrix expectedP(2, 2);
	expectedP(0, 1) = 1.0;
	expectedP(1, 0) = 1.0;

	linalg::Matrix expectedL(2, 2);
	expectedL(0, 0) = 1.0;
	expectedL(1, 0) = 1.0 / 4.0;
	expectedL(1, 1) = 1.0;

	linalg::Matrix expectedU(2, 3);
	expectedU(0, 0) = 4.0;
	expectedU(0, 1) = 5.0;
	expectedU(0, 2) = 6.0;
	expectedU(1, 1) = 3.0 / 4.0;
	expectedU(1, 2) = 3.0 / 2.0;

	return testPLUFactorisation("Wide rectangular PLU", A, expectedP, expectedL, expectedU);
}

// Verifies rectangular factor shapes and multiple pivot swaps for a full-column-rank tall matrix.
static bool testPLUFactorisationTallMatrix() {
	linalg::Matrix A(3, 2);
	A(0, 0) = 1.0;
	A(0, 1) = 2.0;
	A(1, 0) = 2.0;
	A(1, 1) = 4.0;
	A(2, 0) = 3.0;
	A(2, 1) = 7.0;

	linalg::Matrix expectedP(3, 3);
	expectedP(0, 2) = 1.0;
	expectedP(1, 1) = 1.0;
	expectedP(2, 0) = 1.0;

	linalg::Matrix expectedL(3, 3);
	expectedL(0, 0) = 1.0;
	expectedL(1, 0) = 2.0 / 3.0;
	expectedL(1, 1) = 1.0;
	expectedL(2, 0) = 1.0 / 3.0;
	expectedL(2, 1) = 1.0 / 2.0;
	expectedL(2, 2) = 1.0;

	linalg::Matrix expectedU(3, 2);
	expectedU(0, 0) = 3.0;
	expectedU(0, 1) = 7.0;
	expectedU(1, 1) = -2.0 / 3.0;

	return testPLUFactorisation("Tall rectangular PLU", A, expectedP, expectedL, expectedU);
}

// Verifies factors for a rank-deficient square matrix instead of expecting factorisation to throw.
static bool testPLUFactorisationRankDeficientSquareMatrix() {
	linalg::Matrix A(3, 3);
	A(0, 0) = 1.0;
	A(0, 1) = 2.0;
	A(0, 2) = 3.0;
	A(1, 0) = 2.0;
	A(1, 1) = 4.0;
	A(1, 2) = 6.0;
	A(2, 1) = 1.0;
	A(2, 2) = 1.0;

	linalg::Matrix expectedP(3, 3);
	expectedP(0, 1) = 1.0;
	expectedP(1, 2) = 1.0;
	expectedP(2, 0) = 1.0;

	linalg::Matrix expectedL(3, 3);
	expectedL(0, 0) = 1.0;
	expectedL(1, 1) = 1.0;
	expectedL(2, 0) = 1.0 / 2.0;
	expectedL(2, 2) = 1.0;

	linalg::Matrix expectedU(3, 3);
	expectedU(0, 0) = 2.0;
	expectedU(0, 1) = 4.0;
	expectedU(0, 2) = 6.0;
	expectedU(1, 1) = 1.0;
	expectedU(1, 2) = 1.0;

	return testPLUFactorisation("Rank-deficient square PLU", A, expectedP, expectedL, expectedU);
}

// Verifies factors for a rank-deficient wide matrix.
static bool testPLUFactorisationRankDeficientWideMatrix() {
	linalg::Matrix A(2, 3);
	A(0, 0) = 1.0;
	A(0, 1) = 2.0;
	A(0, 2) = 3.0;
	A(1, 0) = 2.0;
	A(1, 1) = 4.0;
	A(1, 2) = 6.0;

	linalg::Matrix expectedP(2, 2);
	expectedP(0, 1) = 1.0;
	expectedP(1, 0) = 1.0;

	linalg::Matrix expectedL(2, 2);
	expectedL(0, 0) = 1.0;
	expectedL(1, 0) = 1.0 / 2.0;
	expectedL(1, 1) = 1.0;

	linalg::Matrix expectedU(2, 3);
	expectedU(0, 0) = 2.0;
	expectedU(0, 1) = 4.0;
	expectedU(0, 2) = 6.0;

	return testPLUFactorisation("Rank-deficient wide PLU", A, expectedP, expectedL, expectedU);
}

// Verifies the independently calculated factors for a denser matrix with several elimination steps.
static bool testPLUFactorisationLargerMatrix() {
	linalg::Matrix A(4, 4);
	A(0, 0) = 2.0;
	A(0, 1) = 1.0;
	A(0, 3) = 1.0;
	A(1, 0) = 2.0;
	A(1, 1) = 1.0;
	A(1, 2) = 2.0;
	A(1, 3) = 3.0;
	A(2, 2) = 1.0;
	A(2, 3) = 2.0;
	A(3, 0) = -4.0;
	A(3, 1) = -1.0;
	A(3, 3) = -2.0;

	linalg::Matrix expectedP(4, 4);
	expectedP(0, 3) = 1.0;
	expectedP(1, 1) = 1.0;
	expectedP(2, 0) = 1.0;
	expectedP(3, 2) = 1.0;

	linalg::Matrix expectedL(4, 4);
	expectedL(0, 0) = 1.0;
	expectedL(1, 0) = -1.0 / 2.0;
	expectedL(1, 1) = 1.0;
	expectedL(2, 0) = -1.0 / 2.0;
	expectedL(2, 1) = 1.0;
	expectedL(2, 2) = 1.0;
	expectedL(3, 0) = 0.0;
	expectedL(3, 1) = 0.0;
	expectedL(3, 2) = -1.0 / 2.0;
	expectedL(3, 3) = 1.0;

	linalg::Matrix expectedU(4, 4);
	expectedU(0, 0) = -4.0;
	expectedU(0, 1) = -1.0;
	expectedU(0, 3) = -2.0;
	expectedU(1, 1) = 1.0 / 2.0;
	expectedU(1, 2) = 2.0;
	expectedU(1, 3) = 2.0;
	expectedU(2, 2) = -2.0;
	expectedU(2, 3) = -2.0;
	expectedU(3, 3) = 1.0;

	return testPLUFactorisation("Larger 4x4 PLU", A, expectedP, expectedL, expectedU);
}

//// Verifies factors for a large 4x4 rank deficient matrix
//static bool testPLUFactorisationLargerRankDeficientMatrix() {
//	linalg::Matrix A(4, 4);
//	
//	A(0, 1) = 2;
//	A(0, 2) = 4;
//	A(0, 3) = 2;
//	
//	A(1, 0) = 2;
//	A(1, 1) = 4;
//	A(1, 2) = 2;
//	A(1, 3) = 6;
//	
//	A(2, 0) = 6;
//	A(2, 1) = 14;
//	A(2, 2) = 10;
//	A(2, 3) = 20;
//
//	A(3, 0) = 4;
//	A(3, 1) = 10;
//	A(3, 2) = 10;
//	A(3, 3) = 13;
//
//	linalg::Matrix expectedP(4, 4);
//	expectedP(0, 1) = 1.0;
//	expectedP(1, 0) = 1.0;
//	expectedP(2, 3) = 1.0;
//	expectedP(3, 2) = 1.0;
//
//	linalg::Matrix expectedL(4, 4);
//	expectedL(0, 0) = 1.0;
//	expectedL(1, 1) = 1.0;
//	expectedL(2, 0) = 2.0;
//	expectedL(2, 1) = 1.0;
//	expectedL(2, 2) = 1.0;
//	expectedL(3, 0) = 3.0;
//	expectedL(3, 1) = 1.0;
//	expectedL(3, 3) = 1.0;
//
//	linalg::Matrix expectedU(4, 4);
//	expectedU(0, 0) = 2;
//	expectedU(0, 1) = 4;
//	expectedU(0, 2) = 2.0;
//	expectedU(0, 3) = 6;
//	expectedU(1, 1) = 2.0;
//	expectedU(1, 2) = 4;
//	expectedU(1, 3) = 2.0;
//	expectedU(2, 2) = 2.0;
//	expectedU(2, 3) = -1.0;
//
//	return testPLUFactorisation("Larger 4x4 PLU", A, expectedP, expectedL, expectedU);
//}


static bool testPLUSolve() {
	linalg::Matrix A(3, 3);

	A(0, 1) = 2;
	A(0, 2) = 1;

	A(1, 0) = 2;
	A(1, 1) = 5;
	A(1, 2) = 4;

	A(2, 0) = 4;
	A(2, 1) = 9;
	A(2, 2) = 9;


	linalg::Matrix b(3, 1);

	b(0, 0) = 1;
	b(1, 0) = 5;
	b(2, 0) = 11;


	linalg::Matrix x(3, 1);

	x(0, 0) = 0.5;
	x(1, 0) = 0;
	x(2, 0) = 1;


	auto factors = A.pluFactorisation();
	auto solution = factors.solve(b);

	std::cout << solution;

	if (solution.isApprox(x)) {
		return true;
	}

	return false;

}


int main() {
	bool allPassed = true;

	// Keep running every test even if an earlier test fails.
	/*allPassed = testPLUFactorisationWithoutPivoting() && allPassed;
	allPassed = testPLUFactorisationWithFirstPivotSwap() && allPassed;
	allPassed = testPLUFactorisationWithMultiplePivotSwaps() && allPassed;
	allPassed = testPLUFactorisationOneByOne() && allPassed;
	allPassed = testPLUFactorisationEmptyMatrix() && allPassed;
	allPassed = testPLUFactorisationWideMatrix() && allPassed;
	allPassed = testPLUFactorisationTallMatrix() && allPassed;
	allPassed = testPLUFactorisationRankDeficientSquareMatrix() && allPassed;
	allPassed = testPLUFactorisationRankDeficientWideMatrix() && allPassed;
	allPassed = testPLUFactorisationLargerMatrix() && allPassed;*/
	

	allPassed = testPLUSolve();

	//allPassed = testPLUFactorisationLargerRankDeficientMatrix() && allPassed;

	// Return the conventional process status: zero means success, nonzero means failure.
	return allPassed ? 0 : 1;
}
