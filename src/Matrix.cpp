#include <linalg/Matrix.hpp>
#include <stdexcept>
#include <ostream>
#include <iomanip>
#include <utility>
#include <algorithm>

namespace linalg {
	Matrix::Matrix(std::size_t rows, std::size_t cols)
		: rows_(rows), cols_(cols), data_(rows * cols, 0.0) {}

	std::size_t Matrix::rows() const { return rows_; }

	std::size_t Matrix::cols() const { return cols_; }

	double& Matrix::operator()(std::size_t i, std::size_t j) {
		if (i >= rows_ || j >= cols_) {
			throw std::out_of_range("Matrix subscript out of range");
		}
		// Row-major indexing formula
		return data_[i * cols_ + j];
	}

	const double& Matrix::operator()(std::size_t i, std::size_t j) const {
		if (i >= rows_ || j >= cols_) {
			throw std::out_of_range("Matrix subscript out of range");
		}
		// Row-major indexing formula
		return data_[i * cols_ + j];
	}

	std::ostream& operator<<(std::ostream& os, const Matrix& A) {
		
		// Save original output stream formatting
		std::streamsize oldPrecision = os.precision();
		std::ios_base::fmtflags oldFlags = os.flags();


		os << std::scientific << std::setprecision(3);

		for (std::size_t i = 0; i < A.rows(); i++) {
			for (std::size_t j = 0; j < A.cols(); j++) {
				os << std::setw(12) << A(i, j) << " ";
			}
			os << '\n';
		}

		// Restore original output stream formatting
		os.precision(oldPrecision);
		os.flags(oldFlags);

		return os;
	}

	Matrix operator+(const Matrix& A, const Matrix& B) {
		if (A.rows() != B.rows() || A.cols() != B.cols()) {
			throw std::invalid_argument("Matrix dimensions don't match.");
		}

		Matrix C(A.rows(), A.cols());

		for (std::size_t i = 0; i < A.rows(); i++) {
			for (std::size_t j = 0; j < A.cols(); j++) {
				C(i, j) = A(i, j) + B(i, j);
			}
		}

		return C;
	}

	Matrix operator*(const Matrix& A, const Matrix& B) {
		if (A.cols() != B.rows()) {
			throw std::invalid_argument("The provided matrices are not multiplicatively conformable.");
		}

		Matrix C(A.rows(), B.cols());

		for (std::size_t i = 0; i < A.rows(); i++) {
			for (std::size_t j = 0; j < B.cols(); j++) {
				for (std::size_t k = 0; k < A.cols(); k++) {
					C(i, j) += A(i, k) * B(k, j);
				}
			}
		}

		return C;
	}

	Matrix operator*(const double s, const Matrix& M) {
		Matrix N(M.rows(), M.cols());

		for (std::size_t i = 0; i < M.rows(); i++) {
			for (std::size_t j = 0; j < M.cols(); j++) {
				N(i, j) = s * M(i, j);
			}
		}

		return N;
	}

	Matrix operator*(const Matrix& M, const double s) {
		return s * M;
	}

	Matrix transpose(const Matrix& M) {
		Matrix N(M.cols(), M.rows());

		for (std::size_t i = 0; i < M.rows(); i++) {
			for (std::size_t j = 0; j < M.cols(); j++) {
				N(j, i) = M(i, j);
			}
		}

		return N;
	}

	Matrix& Matrix::swapRows(const std::size_t row1, const std::size_t row2) {
		if (std::max(row1, row2) >= rows_) {
			throw std::out_of_range("Attempting to access out-of-bounds matrix row.");
		}
		
		for (std::size_t i = 0; i < cols_; i++) {
			std::swap((*this)(row1, i), (*this)(row2, i));
		}

		return (*this);
	}

	Matrix& Matrix::scaleRow(const double scalar, const std::size_t row_number) {
		if (row_number >= rows_) {
			throw std::out_of_range("Attempting to access out-of-bounds matrix row.");
		}

		if (scalar == 0) {
			throw std::invalid_argument("0 is not a valid scalar for row scaling.");
		}

		for (std::size_t i = 0; i < cols_; i++) {
			(*this)(row_number, i) *= scalar;
		}

		return (*this);
	}

	Matrix& Matrix::combineRows(const double scalar, const std::size_t sourceRow, const std::size_t targetRow) {
		if (std::max(sourceRow, targetRow) >= rows_) {
			throw std::out_of_range("Attempting to access out-of-bounds matrix row.");
		}

		if (sourceRow == targetRow) {
			throw std::invalid_argument("Source row cannot also be the target row in a row combination operation.");
		}

		for (std::size_t i = 0; i < cols_; i++) {
			(*this)(targetRow, i) += scalar * (*this)(sourceRow, i);
		}

		return (*this);
	}
}