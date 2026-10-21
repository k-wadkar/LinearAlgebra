#include <linalg/Matrix.hpp>

namespace linalg {
	Matrix::Matrix(std::size_t rows, std::size_t cols)
		: rows_(rows), cols_(cols), data_(rows * cols, 0.0) {}
}