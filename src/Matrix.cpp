#include <linalg/Matrix.hpp>
#include <stdexcept>
#include <ostream>
#include <iomanip>
#include <utility>
#include <algorithm>
#include <iostream>
#include <cmath>

namespace linalg {
	// SECTION: Core matrix construction, dimensions, and element access.
	
	// Matrix constructor definition
	Matrix::Matrix(std::size_t rows, std::size_t cols)
		: rows_(rows), cols_(cols), data_(rows * cols, 0.0) {}

	// Public member function exposing number of rows (read-only)
	std::size_t Matrix::rows() const { return rows_; }

	// Public member function exposing number of rows (read-only)
	std::size_t Matrix::cols() const { return cols_; }

	// Returns the rank of the matrix by performing gaussian elimination on a copy of the receiver
	std::size_t Matrix::rank() const {
		Matrix receiverCopy = (*this);

		// Uses the private member function gaussianElimination on the matrix copy
		// gaussianElimination returns a vector containing the pivot column numbers
		// The number of pivots is equal to the rank, so the size of the vector is the rank
		return receiverCopy.gaussianElimination().size();
	}

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

	Matrix& Matrix::divideRow(const double divisor, const std::size_t row_number) {
		if (row_number >= rows_) {
			throw std::out_of_range("Attempting to access out-of-bounds matrix row.");
		}

		if (divisor == 0 || !std::isfinite(divisor)) {
			throw std::invalid_argument("0 is not a valid scalar for row division.");
		}

		// Apply the same scale to each element in the selected row.
		for (std::size_t i = 0; i < cols_; i++) {
			(*this)(row_number, i) /= divisor;
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

	// (Redundant) Pivot discovery and elimination helpers for echelon-form conversion.
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

	// Swaps with the largest absolute value in the column rather than the first non-zero value.
	// Also referred to as partial pivoting
	std::optional<std::size_t> Matrix::findLargestPivot(const std::size_t selectedCol, const std::size_t startRow) const {
		// Reject out-of-bound arguments
		if (selectedCol >= cols_ || startRow > rows_) {
			throw std::out_of_range("Attempting to access out-of-bounds matrix column or row.");
		}

		// Initialise the return value as there being no suitable pivot value - update when you find one
		std::optional<std::size_t> largestSoFar = std::nullopt;

		// Search only the unprocessed rows so earlier pivot positions remain undisturbed.
		for (std::size_t i = startRow; i < rows_; i++) {
			
			double candidateMagnitude = std::abs((*this)(i, selectedCol));

			// If the selected element is nonzero AND either we haven't found a pivot yet, or the pivots we have found have lower absolute values...
			if ((*this)(i, selectedCol) != 0
				&& (
					!largestSoFar.has_value()
					|| std::abs((*this)(largestSoFar.value(), selectedCol)) < candidateMagnitude)) {
				largestSoFar = i;
			}
		}

		return largestSoFar;
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
			(*this)(targetRow, pivotCol) = 0.0;
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
			auto temp = (*this).findLargestPivot(selectedCol, pivotRow);

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

	// Copies the matrix and performs Gaussian reduction on the copy, then returns it
	Matrix Matrix::echelonForm() const {
		Matrix receiverCopy = (*this);

		receiverCopy.toEchelonForm();

		return receiverCopy;
	}

	// In-place Matrix conversion to RREF.
	Matrix& Matrix::toReducedEchelonForm() {
		std::vector<std::size_t> pivotCols = (*this).gaussianElimination();

		// Iterates backwards through the pivotCols vector
		for (std::size_t i = pivotCols.size(); i > 0; i--) {
			// Contains the value of the currently selected pivot
			double pivotVal = (*this)(i - 1, pivotCols[i - 1]);
			
			// Scales the selected row so the pivot becomes 1
			(*this).divideRow(pivotVal, i - 1);

			// Backwards elimination (makes the entries above in the same column 0)
			for (std::size_t j = 0; j < i - 1; j++) {
				(*this).eliminateEntry(i - 1, pivotCols[i - 1], j);
			}
		}

		return (*this);
	}

	// Copies the matrix and performs Gauss-Jordan reduction on the copy, then returns it
	Matrix Matrix::reducedEchelonForm() const {
		Matrix receiverCopy = (*this);

		receiverCopy.toReducedEchelonForm();

		return receiverCopy;
	}





	// SECTION: Arithmetic and non-mutating matrix transformations.
	Matrix& Matrix::operator+=(const Matrix& other) {
		if (rows() != other.rows() || cols() != other.cols()) {
			throw std::invalid_argument("Matrix dimensions don't match.");
		}

		for (std::size_t i = 0; i < rows(); i++) {
			for (std::size_t j = 0; j < cols(); j++) {
				(*this)(i, j) += other(i, j);
			}
		}

		return *this;
	}

	Matrix& Matrix::operator-=(const Matrix& other) {
		if (rows() != other.rows() || cols() != other.cols()) {
			throw std::invalid_argument("Matrix dimensions don't match.");
		}

		for (std::size_t i = 0; i < rows(); i++) {
			for (std::size_t j = 0; j < cols(); j++) {
				(*this)(i, j) -= other(i, j);
			}
		}

		return *this;
	}

	Matrix& Matrix::operator*=(const Matrix& other) {
		*this = *this * other;
		return *this;
	}

	bool Matrix::isApprox(const Matrix& other, double absTol, double relTol) const {
		if (!std::isfinite(absTol)
			|| !std::isfinite(relTol)
			|| std::min(absTol, relTol) < 0) {
			throw std::invalid_argument("Tolerance values must be non-negative.");
		}
		
		// If the two matrices have different dimensions, they cannot be approximately equivalent
		if ((*this).rows() != other.rows() || (*this).cols() != other.cols()) {
			return false;
		}

		for (std::size_t r = 0; r < (*this).rows(); r++) {
			for (std::size_t c = 0; c < (*this).cols(); c++) {
				double a = (*this)(r, c);
				double b = other(r, c);
				
				// If the two values are exactly equal, skip all further comparisons and move on
				// (includes consideration of infinite values)
				if (a == b) {
					continue;
				}
				// If we are considering this, then the values were not equal
				// In that case, if either value is infinite, we must reject it immediately
				if (!std::isfinite(a) || !std::isfinite(b)) {
					return false;
				}
				
				double absDiff = std::abs(a - b);

				double permittedDiff = absTol + relTol * std::max(std::abs(a), std::abs(b));

				// Normal tolerance formula
				// |a - b| <= eps_abs + eps_rel * max(|a|,|b|)
				if (!(absDiff <= permittedDiff)) {
					return false;
				}
			}
		}

		return true;
	}

	Matrix operator+(const Matrix& A, const Matrix& B) {
		Matrix C = A;
		C += B;
		return C;
	}

	Matrix operator-(const Matrix& A, const Matrix& B) {
		Matrix C = A;
		C -= B;
		return C;
	}

	Matrix operator-(const Matrix& M) {
		return -1.0 * M;
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