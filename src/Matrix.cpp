#include <linalg/Matrix.hpp>
#include <stdexcept>

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
}