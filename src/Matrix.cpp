#include <linalg/Matrix.hpp>
#include <stdexcept>
#include <ostream>
#include <iomanip>

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
			os << "\n";
		}

		// Restore original output stream formatting
		os.precision(oldPrecision);
		os.flags(oldFlags);

		return os;
	}
}