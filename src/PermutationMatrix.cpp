#include <linalg/PermutationMatrix.hpp>
#include <linalg/Matrix.hpp>
#include <numeric>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <algorithm>

namespace linalg {
	PermutationMatrix::PermutationMatrix(std::size_t size) : permutationVector_(size) {
		std::iota(permutationVector_.begin(), permutationVector_.end(), 0);
	}

	std::size_t PermutationMatrix::size() const {
		return permutationVector_.size();
	}

	/*std::vector<std::size_t> PermutationMatrix::permutationVector() const {
		return permutationVector_;
	};*/

	std::size_t PermutationMatrix::originalRow(std::size_t destinationRow) const {
		if (destinationRow >= permutationVector_.size()) {
			throw std::invalid_argument("Attempted to access out-of-bounds row.");
		}
		
		return permutationVector_[destinationRow];
	}

	PermutationMatrix& PermutationMatrix::swapRows(
		std::size_t row1,
		std::size_t row2
	) {
		if (std::max(row1, row2) >= permutationVector_.size()) {
			throw std::out_of_range("Attempting to access out-of-bounds matrix row.");
		}
		
		std::swap(permutationVector_[row1], permutationVector_[row2]);

		return (*this);
	}

	double PermutationMatrix::operator()(std::size_t i, std::size_t j) const {
		if (std::max(i, j) >= permutationVector_.size()) {
			throw std::out_of_range("Attempting to access out-of-bounds matrix row.");
		}
		
		return permutationVector_[i] == j ? 1.0 : 0.0;
	}

	Matrix operator* (const PermutationMatrix& A, const Matrix& B) {
		if (A.size() != B.rows()) {
			throw std::invalid_argument("Attempted multiplication of matrices which are not multiplicatively conformable.");
		}
		
		Matrix C(B.rows(), B.cols());
		
		for (std::size_t i = 0; i < B.rows(); i++) {
			for (std::size_t j = 0; j < B.cols(); j++) {
				C(i, j) = B(A.originalRow(i), j);
			}
			
		}

		return C;
	}


	std::ostream& operator<<(std::ostream& os, const PermutationMatrix& A) {

		// Save original output stream formatting
		std::streamsize oldPrecision = os.precision();
		std::ios_base::fmtflags oldFlags = os.flags();


		os << std::scientific << std::setprecision(3);

		// Print one row per line, padding each entry for aligned columns.
		for (std::size_t i = 0; i < A.size(); i++) {
			for (std::size_t j = 0; j < A.size(); j++) {
				os << std::setw(12) << A(i, j) << " ";
			}
			os << '\n';
		}

		// Restore original output stream formatting
		os.precision(oldPrecision);
		os.flags(oldFlags);

		return os;
	}

}