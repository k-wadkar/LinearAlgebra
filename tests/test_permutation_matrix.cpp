// For testing, use the x64 Developer Command Prompt with working directory LinearAlgebra

// For debug testing, run: 
// cmake --build out/build/x64-debug --target test_permutation_matrix
// ctest --test-dir out/build/x64-debug -V

// For release testing run:
// cmake --build out/build/x64-release --target test_permutation_matrix
// ctest --test-dir out/build/x64-release -V


#include <linalg/PermutationMatrix.hpp>
#include <iostream>

static bool constructorCheck() {
	linalg::PermutationMatrix A(2);
	
	std::cout << "\n\n" << A << "\n\n";

	if (A.size() != 2) {
		std::cerr << "FAILED: Constructor fail - incorrect size.\n";
		
		return false;
	}

	if (A(0, 0) != 1 || A(0, 1) != 0 || A(1, 0) != 0 || A(1, 1) != 1) {
		std::cerr << "FAILED: Constructor fail - wrong entries.\n";

		return false;
	}
	
	std::cout << "PASSED: Basic constructor test\n";
	
	return true;
}

static bool swapCheck() {
	linalg::PermutationMatrix A(2);

	std::cout << "\n\n" << A << "\n\n";

	A.swapRows(0, 1);

	std::cout << "\n\n" << A << "\n\n";

	if (A(0, 0) != 0 || A(0, 1) != 1 || A(1, 0) != 1 || A(1, 1) != 0) {
		std::cerr << "FAILED: Swap fail.\n";

		return false;
	}

	std::cout << "PASSED: Permutation matrix row swap.\n";

	return true;
}





int main() {
	bool allPassed = true;

	// Keep running every test even if an earlier test fails.

	allPassed = constructorCheck() && allPassed;

	allPassed = swapCheck() && allPassed;

	// Return the conventional process status: zero means success, nonzero means failure.
	return allPassed ? 0 : 1;
}