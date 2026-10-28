// For testing, use the x64 Developer Command Prompt with working directory LinearAlgebra

// For debug testing, run:
// cmake --build out/build/x64-debug --target test_permutation_matrix
// ctest --test-dir out/build/x64-debug -V

// For release testing run:
// cmake --build out/build/x64-release --target test_permutation_matrix
// ctest --test-dir out/build/x64-release -V


#include <linalg/PermutationMatrix.hpp>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

// Verifies the size and identity entries of a newly constructed permutation matrix.
static bool testConstructorCreatesIdentity() {
	linalg::PermutationMatrix A(4);

	if (A.size() != 4) {
		std::cerr << "FAILED: Constructor produced the wrong matrix size.\n";
		return false;
	}

	for (std::size_t i = 0; i < A.size(); i++) {
		for (std::size_t j = 0; j < A.size(); j++) {
			const double expected = i == j ? 1.0 : 0.0;
			if (A(i, j) != expected) {
				std::cerr << "FAILED: Constructor did not create an identity matrix.\n";
				return false;
			}
		}
	}

	std::cout << "PASSED: Permutation matrix constructor creates an identity matrix.\n";
	return true;
}

// Verifies that constructing a zero-size permutation matrix is valid.
static bool testZeroSizeMatrix() {
	linalg::PermutationMatrix A(0);

	if (A.size() != 0) {
		std::cerr << "FAILED: Zero-size matrix reported a nonzero size.\n";
		return false;
	}

	std::cout << "PASSED: Zero-size permutation matrix.\n";
	return true;
}

// Verifies that row swaps move the corresponding identity rows into the expected order.
static bool testSwapRows() {
	linalg::PermutationMatrix A(3);

	A.swapRows(0, 2);

	const double expected[3][3] = {
		{0.0, 0.0, 1.0},
		{0.0, 1.0, 0.0},
		{1.0, 0.0, 0.0}
	};

	for (std::size_t i = 0; i < A.size(); i++) {
		for (std::size_t j = 0; j < A.size(); j++) {
			if (A(i, j) != expected[i][j]) {
				std::cerr << "FAILED: Row swap produced incorrect matrix entries.\n";
				return false;
			}
		}
	}

	std::cout << "PASSED: Permutation matrix row swap.\n";
	return true;
}

// Verifies that swapping a row with itself leaves the matrix unchanged.
static bool testSwapRowWithItself() {
	linalg::PermutationMatrix A(3);

	A.swapRows(1, 1);

	for (std::size_t i = 0; i < A.size(); i++) {
		for (std::size_t j = 0; j < A.size(); j++) {
			const double expected = i == j ? 1.0 : 0.0;
			if (A(i, j) != expected) {
				std::cerr << "FAILED: Swapping a row with itself changed the matrix.\n";
				return false;
			}
		}
	}

	std::cout << "PASSED: Swapping a row with itself.\n";
	return true;
}

// Verifies that row swaps can be chained and return a reference to the original matrix.
static bool testSwapRowsCanBeChained() {
	linalg::PermutationMatrix A(3);
	linalg::PermutationMatrix& returned = A.swapRows(0, 2).swapRows(1, 2);

	if (&returned != &A) {
		std::cerr << "FAILED: Row swap did not return a reference to the matrix.\n";
		return false;
	}

	const double expected[3][3] = {
		{0.0, 0.0, 1.0},
		{1.0, 0.0, 0.0},
		{0.0, 1.0, 0.0}
	};

	for (std::size_t i = 0; i < A.size(); i++) {
		for (std::size_t j = 0; j < A.size(); j++) {
			if (A(i, j) != expected[i][j]) {
				std::cerr << "FAILED: Chained row swaps produced incorrect entries.\n";
				return false;
			}
		}
	}

	std::cout << "PASSED: Chained permutation matrix row swaps.\n";
	return true;
}

// Verifies that an invalid first row index throws without changing the matrix.
static bool testSwapRowsRejectsInvalidFirstRow() {
	linalg::PermutationMatrix A(2);
	bool threwOutOfRange = false;

	try {
		A.swapRows(2, 0);
	}
	catch (const std::out_of_range&) {
		threwOutOfRange = true;
	}

	if (!threwOutOfRange || A(0, 0) != 1.0 || A(0, 1) != 0.0
		|| A(1, 0) != 0.0 || A(1, 1) != 1.0) {
		std::cerr << "FAILED: Invalid first row was not rejected safely.\n";
		return false;
	}

	std::cout << "PASSED: Row swap rejects an invalid first row.\n";
	return true;
}

// Verifies that an invalid second row index throws without changing the matrix.
static bool testSwapRowsRejectsInvalidSecondRow() {
	linalg::PermutationMatrix A(2);
	bool threwOutOfRange = false;

	try {
		A.swapRows(0, 2);
	}
	catch (const std::out_of_range&) {
		threwOutOfRange = true;
	}

	if (!threwOutOfRange || A(0, 0) != 1.0 || A(0, 1) != 0.0
		|| A(1, 0) != 0.0 || A(1, 1) != 1.0) {
		std::cerr << "FAILED: Invalid second row was not rejected safely.\n";
		return false;
	}

	std::cout << "PASSED: Row swap rejects an invalid second row.\n";
	return true;
}

// Verifies that reading an entry with an invalid row index throws.
static bool testEntryAccessRejectsInvalidRow() {
	linalg::PermutationMatrix A(2);
	bool threwOutOfRange = false;

	try {
		(void)A(2, 0);
	}
	catch (const std::out_of_range&) {
		threwOutOfRange = true;
	}

	if (!threwOutOfRange) {
		std::cerr << "FAILED: Entry access accepted an invalid row index.\n";
		return false;
	}

	std::cout << "PASSED: Entry access rejects an invalid row index.\n";
	return true;
}

// Verifies that reading an entry with an invalid column index throws.
static bool testEntryAccessRejectsInvalidColumn() {
	linalg::PermutationMatrix A(2);
	bool threwOutOfRange = false;

	try {
		(void)A(0, 2);
	}
	catch (const std::out_of_range&) {
		threwOutOfRange = true;
	}

	if (!threwOutOfRange) {
		std::cerr << "FAILED: Entry access accepted an invalid column index.\n";
		return false;
	}

	std::cout << "PASSED: Entry access rejects an invalid column index.\n";
	return true;
}

// Verifies that indexing an empty permutation matrix throws.
static bool testEmptyMatrixEntryAccessThrows() {
	linalg::PermutationMatrix A(0);
	bool threwOutOfRange = false;

	try {
		(void)A(0, 0);
	}
	catch (const std::out_of_range&) {
		threwOutOfRange = true;
	}

	if (!threwOutOfRange) {
		std::cerr << "FAILED: Empty matrix entry access did not throw.\n";
		return false;
	}

	std::cout << "PASSED: Empty matrix entry access throws.\n";
	return true;
}

// Verifies the formatted matrix output and that stream formatting is restored afterward.
static bool testStreamOutputAndFormattingState() {
	linalg::PermutationMatrix A(2);
	A.swapRows(0, 1);

	std::ostringstream output;
	output << std::fixed << std::setprecision(1);
	const std::ios_base::fmtflags originalFlags = output.flags();
	const std::streamsize originalPrecision = output.precision();

	std::ostream& returned = output << A;

	const std::string expected =
		"   0.000e+00    1.000e+00 \n"
		"   1.000e+00    0.000e+00 \n";

	if (&returned != &output || output.str() != expected) {
		std::cerr << "FAILED: Permutation matrix stream output was incorrect.\n";
		return false;
	}

	if (output.flags() != originalFlags || output.precision() != originalPrecision) {
		std::cerr << "FAILED: Stream output did not restore formatting state.\n";
		return false;
	}

	output << ' ' << 1.26;
	if (output.str().substr(expected.size()) != " 1.3") {
		std::cerr << "FAILED: Stream formatting was not preserved after matrix output.\n";
		return false;
	}

	std::cout << "PASSED: Permutation matrix stream output and formatting state.\n";
	return true;
}

// Verifies that streaming an empty matrix produces no output.
static bool testEmptyMatrixStreamOutput() {
	linalg::PermutationMatrix A(0);
	std::ostringstream output;

	output << A;

	if (!output.str().empty()) {
		std::cerr << "FAILED: Empty matrix stream output was not empty.\n";
		return false;
	}

	std::cout << "PASSED: Empty permutation matrix stream output.\n";
	return true;
}

int main() {
	bool allPassed = true;

	// Keep running every test even if an earlier test fails.
	allPassed = testConstructorCreatesIdentity() && allPassed;
	allPassed = testZeroSizeMatrix() && allPassed;
	allPassed = testSwapRows() && allPassed;
	allPassed = testSwapRowWithItself() && allPassed;
	allPassed = testSwapRowsCanBeChained() && allPassed;
	allPassed = testSwapRowsRejectsInvalidFirstRow() && allPassed;
	allPassed = testSwapRowsRejectsInvalidSecondRow() && allPassed;
	allPassed = testEntryAccessRejectsInvalidRow() && allPassed;
	allPassed = testEntryAccessRejectsInvalidColumn() && allPassed;
	allPassed = testEmptyMatrixEntryAccessThrows() && allPassed;
	allPassed = testStreamOutputAndFormattingState() && allPassed;
	allPassed = testEmptyMatrixStreamOutput() && allPassed;

	// Return the conventional process status: zero means success, nonzero means failure.
	return allPassed ? 0 : 1;
}
