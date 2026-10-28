#pragma once

#include <linalg/Matrix.hpp>
#include <linalg/PermutationMatrix.hpp>

namespace linalg {

	struct PLUFactorisation {
		PermutationMatrix P;
		Matrix L;
		Matrix U;
	};
}