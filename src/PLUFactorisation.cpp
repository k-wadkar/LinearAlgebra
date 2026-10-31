#include <linalg/PLUFactorisation.hpp>
#include <stdexcept>
#include <cmath>

namespace linalg {
	
	// Assumes L is UNIT lower triangular
	static Matrix forwardSubstitution(
		const Matrix& L,
		const Matrix& b
	) {
		if (L.rows() != L.cols()) {
			throw std::invalid_argument("L is not a square matrix.");
		}
		else if (L.rows() != b.rows()) {
			throw std::invalid_argument("L and b have incompatible dimensions for forward substution.");
		}
		else if (b.cols() != 1) {
			throw std::invalid_argument("b is not a column vector.");
		}
		
		linalg::Matrix y(L.rows(), 1);

		for (std::size_t i = 0; i < y.rows(); i++) {
			double acc = 0;

			for (std::size_t j = 0; j < i; j++) {
				acc = std::fma(L(i, j), y(j, 0), acc);
			}
			y(i, 0) = b(i, 0) - acc;
		}

		return y;
	}


	static Matrix backwardSubstitution(
		const Matrix& U,
		const Matrix& y
	) {
		if (U.rows() != U.cols()) {
			throw std::invalid_argument("U is not a square matrix.");
		}
		else if (U.rows() != y.rows()) {
			throw std::invalid_argument("U and y have incompatible dimensions for backward substution.");
		}
		else if (y.cols() != 1) {
			throw std::invalid_argument("y is not a column vector.");
		}

		linalg::Matrix x(U.rows(), 1);

		for (std::size_t i = x.rows(); i > 0; i--) {
			double acc = 0;

			for (std::size_t j = x.rows(); j > i; j--) {
				acc = std::fma(U(i-1, j-1), x(j-1, 0), acc);
			}
			if (U(i - 1, i - 1) == 0.0) {
				throw std::invalid_argument("Infinite number of solutions.");
			}
			
			x(i - 1, 0) = (y(i - 1, 0) - acc) / U(i-1,i-1);
		}

		return x;
	}

	
	Matrix PLUFactorisation::solve(const Matrix& b) const {
		if (b.rows() != (*this).P.size() || b.cols() != 1) {
			throw std::invalid_argument("b has the wrong dimension.");
		}
		
		Matrix y = forwardSubstitution((*this).L, (*this).P * b);

		return backwardSubstitution((*this).U, y);;
	}
}