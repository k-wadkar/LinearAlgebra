#pragma once

#include <vector>
#include <cstddef>

namespace linalg {
	class Matrix {
	private:
		std::size_t rows_;
		std::size_t cols_;
		std::vector<double> data_;

	public:
		Matrix(std::size_t rows, std::size_t cols);
	};
}