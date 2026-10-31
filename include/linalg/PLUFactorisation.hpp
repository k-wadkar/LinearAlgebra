#pragma once

#include <linalg/Matrix.hpp>
#include <linalg/PermutationMatrix.hpp>

namespace linalg {

	struct PLUFactorisation {
		PermutationMatrix P;
		Matrix L;
		Matrix U;

		Matrix solve(const Matrix& b) const;
	};

	static Matrix forwardSubstitution(
		const Matrix& L,
		const Matrix& b
	);

	static Matrix backwardSubstitution(
		const Matrix& U,
		const Matrix& y
	);
}