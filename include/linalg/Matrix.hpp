#pragma once

#include <vector>
#include <cstddef>
#include <iosfwd>

namespace linalg {
	class Matrix {
	private:
		// Member variables to store the number of rows, columns, and the data of the matrix
		// size_t is used for sizes and indices as it is an unsigned integer type
		// The data is stored in a 1D vector, and the elements can be accessed using row-major order
		std::size_t rows_;
		std::size_t cols_;
		std::vector<double> data_;

	public:
		// Constructor to initialize the matrix with given number of rows and columns
		Matrix(std::size_t rows, std::size_t cols);
		
		// Member functions to get the number of rows and columns
		std::size_t rows() const;
		std::size_t cols() const;

		// Overloaded operator() to access matrix elements in a 2D manner
		double& operator()(std::size_t i, std::size_t j);

		const double& operator()(std::size_t i, std::size_t j) const;
	};

	std::ostream& operator<<(std::ostream& os, const Matrix& A);

	Matrix operator+(const Matrix& A, const Matrix& B);

	Matrix operator*(const Matrix& A, const Matrix& B);

	Matrix operator*(const double s, const Matrix& M);

	Matrix operator*(const Matrix& M, const double s);
}