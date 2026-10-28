#pragma once

#include <vector>
#include <cstddef>
#include <iosfwd>
#include <optional>

namespace linalg {
	
	class PermutationMatrix;
	
	// A dynamically sized matrix of double-precision values.
	class Matrix {
	private:
		// Dimensions and flat row-major storage; size_t matches container sizes and indices.
		std::size_t rows_;
		std::size_t cols_;
		std::vector<double> data_;

		// Returns the first nonzero row at or below startRow, or nullopt if none exists.
		// optional distinguishes "no pivot found" from a valid row index without a sentinel.
		// (A sentinel is a special value used to signal a condition, commonly the absence of a valid result.)
		// Column and row indices use size_t to match the matrix dimensions.
		std::optional<std::size_t> findFirstPivot (const std::size_t selectedCol, const std::size_t startRow) const;

		// Returns the row at or below startRow which contains the element with the greatest non-zero absolute value in selectedCol
		// optional distinguished "no pivot found" from a valid row index without a sentinel
		std::optional<std::size_t> findLargestPivot(
			const std::size_t selectedCol, 
			const std::size_t startRow, 
			double absolutePivotTheshold
		) const;

		// Uses the pivot row to zero the selected column in targetRow; modifies this matrix.
		double eliminateEntry(std::size_t pivotRow, std::size_t pivotCol, std::size_t targetRow);

		// Private helper to scaleRow, except divides the row by the scalar rather than multiplying it.
		// Helpful for improving numerical stability.
		Matrix& divideRow(const double divisor, const std::size_t row_number);

		// Performs Gaussian elimination and returns the pivot column numbers in a vector
		std::vector<std::size_t> gaussianElimination(
			double relTol = 0.0,
			Matrix* L = nullptr,
			PermutationMatrix* P = nullptr
		);

	public:
		// Creates a rows-by-cols matrix initialized with zeroes.
		Matrix(std::size_t rows, std::size_t cols);

		// Return dimensions by value; the small integer values need no reference or copy avoidance.
		std::size_t rows() const;
		std::size_t cols() const;

		std::size_t rank() const;

		// Mutable overload returns a reference so callers can assign to an element.
		double& operator()(std::size_t i, std::size_t j);

		// Const overload permits element reads from a const matrix without allowing modification.
		const double& operator()(std::size_t i, std::size_t j) const;

		// Compare with another matrix by const reference to avoid copying; defaulted equality
		// checks dimensions and stored values member by member.
		bool operator== (const Matrix& other) const = default;


		bool isApprox(const Matrix& other, double absTol = 1e-12, double relTol = 1e-9) const;


		// Compound arithmetic modifies this matrix and returns it for chaining.
		Matrix& operator+=(const Matrix& other);
		Matrix& operator-=(const Matrix& other);
		Matrix& operator*=(const Matrix& other);

		// Row indices use size_t to match dimensions; operations return this matrix by reference
		// so calls can be chained without copying the matrix.
		Matrix& swapRows(const std::size_t row1, const std::size_t row2);

		// The double scalar represents a numeric scale; row_number is a size_t row index.
		Matrix& scaleRow(const double scalar, const std::size_t row_number);

		// Add scalar times sourceRow to targetRow; double represents the factor and size_t the indices.
		Matrix& combineRows(const double scalar, const std::size_t sourceRow, const std::size_t targetRow);

		// (Mutating) Calls gaussianElimination to perform forward Gaussian elimination in place, and return the matrix for chaining.
		Matrix& toEchelonForm(double relTol = 0.0);

		// (Non-mutating) Calls toEchelonForm on a copy of the calling object Matrix and returns the copy by value
		Matrix echelonForm(double relTol = 0.0) const;


		// (Mutating) Calls gaussianElimination to get pivot locations, and performs backwards elimination to matrix in RREF
		Matrix& toReducedEchelonForm(double relTol = 0.0);

		// (Non-mutating) Calls toReducedEchelonForm on a copy of the calling object and returns the copy by value
		Matrix reducedEchelonForm(double relTol = 0.0) const;

		// Returns the greatest absolute value in the matrix
		double maxAbsoluteRealVal() const;
	};

	// Write a matrix to the supplied stream and return the stream to support chained output.
	std::ostream& operator<<(std::ostream& os, const Matrix& A);

	// Arithmetic operators take const references to avoid copying inputs and return new results.
	Matrix operator+(const Matrix& A, const Matrix& B);
	Matrix operator-(const Matrix& A, const Matrix& B);
	Matrix operator-(const Matrix& M);

	Matrix operator*(const Matrix& A, const Matrix& B);

	// Both scalar-multiplication overloads support scalar * matrix and matrix * scalar syntax.
	Matrix operator*(const double s, const Matrix& M);

	Matrix operator*(const Matrix& M, const double s);

	// Return a new matrix with rows and columns exchanged.
	Matrix transpose(const Matrix& M);
}