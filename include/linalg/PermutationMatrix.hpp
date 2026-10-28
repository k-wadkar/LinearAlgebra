#pragma once

#include <cstddef>
#include <vector>
#include <iosfwd>

namespace linalg {

	class PermutationMatrix {
	private:
		std::vector<std::size_t> permutationVector_;
	public:
		// 
		PermutationMatrix(std::size_t size);

		std::size_t size() const;

		PermutationMatrix& swapRows(std::size_t row1, std::size_t row2);

		double operator()(std::size_t i, std::size_t j) const;
	};

	// Write a permutation matrix to the supplied stream and return the stream to support chained output.
	std::ostream& operator<<(std::ostream& os, const PermutationMatrix& A);
}