#include <linalg/Matrix.hpp>
#include <stdexcept>
#include <ostream>
#include <iomanip>
#include <utility>
#include <algorithm>
#include <iostream>

namespace linalg {
	// SECTION: Core matrix construction, dimensions, and element access.
	
	// Matrix constructor definition
	Matrix::Matrix(std::size_t rows, std::size_t cols)
		: rows_(rows), cols_(cols), data_(rows * cols, 0.0) {}

	// Public member function exposing number of rows (read-only)
	std::size_t Matrix::rows() const { return rows_; }

	// Public member function exposing number of rows (read-only)
	std::size_t Matrix::cols() const { return cols_; }

	// The non-const overload returns a writable reference to the stored element.
	double& Matrix::operator()(std::size_t i, std::size_t j) {
		if (i >= rows_ || j >= cols_) {
			throw std::out_of_range("Matrix subscript out of range");
		}
		// Row-major indexing maps each row to one contiguous span in data_.
		return data_[i * cols_ + j];
	}

	// The const overload uses the same bounds check and indexing but exposes a read-only reference.
	const double& Matrix::operator()(std::size_t i, std::size_t j) const {
		if (i >= rows_ || j >= cols_) {
			throw std::out_of_range("Matrix subscript out of range");
		}
		return data_[i * cols_ + j];
	}





	// SECTION: Elementary row operations used by Gaussian elimination.
	
	Matrix& Matrix::swapRows(const std::size_t row1, const std::size_t row2) {
		if (std::max(row1, row2) >= rows_) {
			throw std::out_of_range("Attempting to access out-of-bounds matrix row.");
		}
		
		// Swap every corresponding column entry to exchange the complete rows.
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

		// Apply the same scale to each element in the selected row.
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

		// Update each target entry using the matching source entry.
		for (std::size_t i = 0; i < cols_; i++) {
			(*this)(targetRow, i) += scalar * (*this)(sourceRow, i);
		}

		return (*this);
	}

	// Pivot discovery and elimination helpers for echelon-form conversion.
	std::optional<std::size_t> Matrix::findFirstPivot(const std::size_t selectedCol, const std::size_t startRow) const {
		// startRow == rows_ is a valid empty search range; values beyond it are invalid.
		if (selectedCol >= cols_ || startRow > rows_) {
			throw std::out_of_range("Attempting to access out-of-bounds matrix column or row.");
		}
		
		// Search only the unprocessed rows so earlier pivot positions remain undisturbed.
		for (std::size_t i = startRow; i < rows_; i++) {
			if ((*this)(i, selectedCol) != 0) {
				return i;
			}
		}

		// No suitable row exists in this column at or below startRow.
		return std::nullopt;
	}

	// Uses row combination to eliminate a specific entry (i.e. make it zero)
	// Numerically naive, so may actually produce a tiny value rather than exactly zero
	void Matrix::eliminateEntry(std::size_t pivotRow, std::size_t pivotCol, std::size_t targetRow) {
		if ((*this)(pivotRow, pivotCol) == 0) {
			throw std::invalid_argument("Attempted to eliminate entry using a zero pivot.");
		}
		
		if ((*this)(targetRow, pivotCol) != 0) { // Need to change this later to work with tolerance
			double scalar = -(*this)(targetRow, pivotCol) / (*this)(pivotRow, pivotCol);
			(*this).combineRows(scalar, pivotRow, targetRow);
		}
	}

	// (Private) In-place Matrix conversion to Echelon form.
	// Returns a vector containing pivot column numbers
	std::vector<std::size_t> Matrix::gaussianElimination() {
		std::vector<std::size_t> pivotCols;
		std::size_t pivotRow = 0;
		std::size_t selectedCol = 0;
		
		// Advance through columns, skipping columns without a pivot and advancing rows only when one is found.
		while (pivotRow < rows_ && selectedCol < cols_) {
			auto temp = (*this).findFirstPivot(selectedCol, pivotRow);

			// If there's a pivot in this column...
			if (temp.has_value()) {
				pivotCols.push_back(selectedCol);
				
				// Keep the current row when possible; otherwise bring the discovered pivot into position.
				if (pivotRow != temp.value()) {
					(*this).swapRows(pivotRow, temp.value());
				}

				// Clear entries below this pivot so each subsequent row has an earlier zero prefix.
				for (std::size_t i = pivotRow + 1; i < rows_; i++) {
					(*this).eliminateEntry(pivotRow, selectedCol, i);
				}

				pivotRow++;

			}

			selectedCol++;

		}

		return pivotCols;
	}

	// In-place Matrix conversion to Echelon Form.
	Matrix& Matrix::toEchelonForm() {
		(*this).gaussianElimination();

		return (*this);
	}

	// In-place Matrix conversion to RREF.
	Matrix& Matrix::toReducedEchelonForm() {
		std::vector<std::size_t> pivotCols = (*this).gaussianElimination();

		// Iterates backwards through the pivotCols vector
		for (std::size_t i = pivotCols.size(); i > 0; i--) {
			// Contains the value of the currently selected pivot
			double pivotVal = (*this)(i - 1, pivotCols[i - 1]);
			
			// Scales the selected row so the pivot becomes 1
			(*this).scaleRow(1 / pivotVal, i - 1);

			// Backwards elimination (makes the entries above in the same column 0)
			for (std::size_t j = 0; j < i - 1; j++) {
				(*this).eliminateEntry(i - 1, pivotCols[i - 1], j);
			}
		}

		return (*this);
	}





	// SECTION: Arithmetic and non-mutating matrix transformations.
	Matrix operator+(const Matrix& A, const Matrix& B) {
		if (A.rows() != B.rows() || A.cols() != B.cols()) {
			throw std::invalid_argument("Matrix dimensions don't match.");
		}

		Matrix C(A.rows(), A.cols());

		// Each output entry is the sum of the entries at the same position.
		for (std::size_t i = 0; i < A.rows(); i++) {
			for (std::size_t j = 0; j < A.cols(); j++) {
				C(i, j) = A(i, j) + B(i, j);
			}
		}

		return C;
	}

	/* Three overloads for operator*:
		- One for matrix-matrix multiplication
		- One for scalar-matrix multiplication
		- One for matrix-scalar multiplication
	 The order of arguments matters, so we need to account for both scalar multiplication orders. */
	Matrix operator*(const Matrix& A, const Matrix& B) {
		if (A.cols() != B.rows()) {
			throw std::invalid_argument("The provided matrices are not multiplicatively conformable.");
		}

		Matrix C(A.rows(), B.cols());

		// Each output entry is a dot product: k traverses a row of A and column of B.
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

		// Scalar multiplication creates a result, leaving the input matrix unchanged.
		for (std::size_t i = 0; i < M.rows(); i++) {
			for (std::size_t j = 0; j < M.cols(); j++) {
				N(i, j) = s * M(i, j);
			}
		}

		return N;
	}

	Matrix operator*(const Matrix& M, const double s) {
		// Reuse the scalar-left overload so both operand orders have identical behavior.
		return s * M;
	}

	// This is a free function so returns a value rather than modifying the input matrix in-place
	Matrix transpose(const Matrix& M) {
		Matrix N(M.cols(), M.rows());

		// Assign each source element to the position with row and column exchanged.
		for (std::size_t i = 0; i < M.rows(); i++) {
			for (std::size_t j = 0; j < M.cols(); j++) {
				N(j, i) = M(i, j);
			}
		}

		return N;
	}

	
	
	
	
	// SECTION: Output

	// Stream output uses a fixed display format but restores the caller's stream state afterward.
	std::ostream& operator<<(std::ostream& os, const Matrix& A) {

		// Save original output stream formatting
		std::streamsize oldPrecision = os.precision();
		std::ios_base::fmtflags oldFlags = os.flags();


		os << std::scientific << std::setprecision(3);

		// Print one row per line, padding each entry for aligned columns.
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

}