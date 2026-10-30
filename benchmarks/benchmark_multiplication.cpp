#include <Eigen/Dense>
#include <linalg/Matrix.hpp>
#include <benchmark/benchmark.h>
#include <stdexcept>
//#include <iostream>

using MultiplicationFunction = linalg::Matrix (*)(
	const linalg::Matrix&,
	const linalg::Matrix&
);



//NUM 1
static linalg::Matrix multiplyIJK(
	const linalg::Matrix& A,
	const linalg::Matrix& B
) {
	if (A.cols() != B.rows()) {
		throw std::invalid_argument("The provided matrices are not multiplicatively conformable.");
	}

	linalg::Matrix C(A.rows(), B.cols());

	for (std::size_t i = 0; i < A.rows(); i++) {
		for (std::size_t j = 0; j < B.cols(); j++) {
			for (std::size_t k = 0; k < A.cols(); k++) {
				C(i, j) += A(i, k) * B(k, j);
			}
		}
	}

	return C;
}

//NUM 2
static linalg::Matrix multiplyIKJ(
	const linalg::Matrix& A,
	const linalg::Matrix& B
) {
	if (A.cols() != B.rows()) {
		throw std::invalid_argument("The provided matrices are not multiplicatively conformable.");
	}

	linalg::Matrix C(A.rows(), B.cols());

	for (std::size_t i = 0; i < A.rows(); i++) {
		for (std::size_t k = 0; k < A.cols(); k++) {
			for (std::size_t j = 0; j < B.cols(); j++) {
				C(i, j) += A(i, k) * B(k, j);
			}
		}
	}

	return C;
}

//NUM 3
static linalg::Matrix multiplyJIK(
	const linalg::Matrix& A,
	const linalg::Matrix& B
) {
	if (A.cols() != B.rows()) {
		throw std::invalid_argument("The provided matrices are not multiplicatively conformable.");
	}

	linalg::Matrix C(A.rows(), B.cols());

	for (std::size_t j = 0; j < B.cols(); j++) {
		for (std::size_t i = 0; i < A.rows(); i++) {
			for (std::size_t k = 0; k < A.cols(); k++) {		
				C(i, j) += A(i, k) * B(k, j);
			}
		}
	}

	return C;
}

//NUM 4
static linalg::Matrix multiplyJKI(
	const linalg::Matrix& A,
	const linalg::Matrix& B
) {
	if (A.cols() != B.rows()) {
		throw std::invalid_argument("The provided matrices are not multiplicatively conformable.");
	}

	linalg::Matrix C(A.rows(), B.cols());

	for (std::size_t j = 0; j < B.cols(); j++) {
		for (std::size_t k = 0; k < A.cols(); k++) {
			for (std::size_t i = 0; i < A.rows(); i++) {
				C(i, j) += A(i, k) * B(k, j);
			}
		}
	}

	return C;
}


//NUM 5
static linalg::Matrix multiplyKIJ(
	const linalg::Matrix& A,
	const linalg::Matrix& B
) {
	if (A.cols() != B.rows()) {
		throw std::invalid_argument("The provided matrices are not multiplicatively conformable.");
	}

	linalg::Matrix C(A.rows(), B.cols());

	for (std::size_t k = 0; k < A.cols(); k++) {		
		for (std::size_t i = 0; i < A.rows(); i++) {
			for (std::size_t j = 0; j < B.cols(); j++) {
				C(i, j) += A(i, k) * B(k, j);
			}
		}
	}

	return C;
}

//NUM 6
static linalg::Matrix multiplyKJI(
	const linalg::Matrix& A,
	const linalg::Matrix& B
) {
	if (A.cols() != B.rows()) {
		throw std::invalid_argument("The provided matrices are not multiplicatively conformable.");
	}

	linalg::Matrix C(A.rows(), B.cols());

	for (std::size_t k = 0; k < A.cols(); k++) {
		for (std::size_t j = 0; j < B.cols(); j++) {
			for (std::size_t i = 0; i < A.rows(); i++) {
				C(i, j) += A(i, k) * B(k, j);
			}
		}
	}

	return C;
}



static void BM_MultiplicationIJK(benchmark::State& state) {
	const std::size_t n = static_cast<std::size_t>(state.range(0));
	
	linalg::Matrix A(n, n);
	linalg::Matrix B(n, n);

	// Deterministic matrix content
	for (std::size_t i = 0; i < n; i++) {
		for (std::size_t j = 0; j < n; j++) {
			A(i, j) = static_cast<double>(i + j + 1);
			B(i, j) = static_cast<double>(i + 2 * j + 1);
		}
	}

	for (auto _ : state) {
		linalg::Matrix C = multiplyIJK(A, B);
		benchmark::DoNotOptimize(C);
	}
}

static void BM_MultiplicationIKJ(benchmark::State& state) {
	const std::size_t n = static_cast<std::size_t>(state.range(0));
	linalg::Matrix A(n, n);
	linalg::Matrix B(n, n);

	// Deterministic matrix content
	for (std::size_t i = 0; i < n; i++) {
		for (std::size_t j = 0; j < n; j++) {
			A(i, j) = static_cast<double>(i + j + 1);
			B(i, j) = static_cast<double>(i + 2 * j + 1);
		}
	}

	for (auto _ : state) {
		linalg::Matrix C = multiplyIKJ(A, B);
		benchmark::DoNotOptimize(C);
	}
}

static void BM_MultiplicationJIK(benchmark::State& state) {
	const std::size_t n = static_cast<std::size_t>(state.range(0));
	linalg::Matrix A(n, n);
	linalg::Matrix B(n, n);

	// Deterministic matrix content
	for (std::size_t i = 0; i < n; i++) {
		for (std::size_t j = 0; j < n; j++) {
			A(i, j) = static_cast<double>(i + j + 1);
			B(i, j) = static_cast<double>(i + 2 * j + 1);
		}
	}

	for (auto _ : state) {
		linalg::Matrix C = multiplyJIK(A, B);
		benchmark::DoNotOptimize(C);
	}
}

static void BM_MultiplicationJKI(benchmark::State& state) {
	const std::size_t n = static_cast<std::size_t>(state.range(0));
	linalg::Matrix A(n, n);
	linalg::Matrix B(n, n);

	// Deterministic matrix content
	for (std::size_t i = 0; i < n; i++) {
		for (std::size_t j = 0; j < n; j++) {
			A(i, j) = static_cast<double>(i + j + 1);
			B(i, j) = static_cast<double>(i + 2 * j + 1);
		}
	}

	for (auto _ : state) {
		linalg::Matrix C = multiplyJKI(A, B);
		benchmark::DoNotOptimize(C);
	}
}

static void BM_MultiplicationKIJ(benchmark::State& state) {
	const std::size_t n = static_cast<std::size_t>(state.range(0));
	linalg::Matrix A(n, n);
	linalg::Matrix B(n, n);

	// Deterministic matrix content
	for (std::size_t i = 0; i < n; i++) {
		for (std::size_t j = 0; j < n; j++) {
			A(i, j) = static_cast<double>(i + j + 1);
			B(i, j) = static_cast<double>(i + 2 * j + 1);
		}
	}

	for (auto _ : state) {
		linalg::Matrix C = multiplyKIJ(A, B);
		benchmark::DoNotOptimize(C);
	}
}

static void BM_MultiplicationKJI(benchmark::State& state) {
	const std::size_t n = static_cast<std::size_t>(state.range(0));
	linalg::Matrix A(n, n);
	linalg::Matrix B(n, n);

	// Deterministic matrix content
	for (std::size_t i = 0; i < n; i++) {
		for (std::size_t j = 0; j < n; j++) {
			A(i, j) = static_cast<double>(i + j + 1);
			B(i, j) = static_cast<double>(i + 2 * j + 1);
		}
	}

	for (auto _ : state) {
		linalg::Matrix C = multiplyKJI(A, B);
		benchmark::DoNotOptimize(C);
	}
}



static void BM_EigenMultiplication(benchmark::State& state) {
	Eigen::MatrixXd A(100, 100);
	Eigen::MatrixXd B(100, 100);
	
	// Deterministic matrix content
	for (std::size_t i = 0; i < 100; i++) {
		for (std::size_t j = 0; j < 100; j++) {
			A(i, j) = static_cast<double>(i + j + 1);
			B(i, j) = static_cast<double>(i + 2 * j + 1);
		}
	}

	for (auto _ : state) {
		Eigen::MatrixXd C = A * B;
		benchmark::DoNotOptimize(C);
	}
}

static void BM_LinalgMultiplication(benchmark::State& state) {
	linalg::Matrix A(100, 100);
	linalg::Matrix B(100, 100);

	// Deterministic matrix content
	for (std::size_t i = 0; i < 100; i++) {
		for (std::size_t j = 0; j < 100; j++) {
			A(i, j) = static_cast<double>(i + j + 1);
			B(i, j) = static_cast<double>(i + 2 * j + 1);
		}
	}

	for (auto _ : state) {
		linalg::Matrix C = A * B;
		benchmark::DoNotOptimize(C);
	}
}

BENCHMARK(BM_EigenMultiplication);
BENCHMARK(BM_LinalgMultiplication);

//BENCHMARK(BM_MultiplicationIJK)
//->Arg(10)
//->Arg(50)
//->Arg(100)
//->Arg(250)
//->Arg(500);
BENCHMARK(BM_MultiplicationIKJ)
->Arg(10)
->Arg(50)
->Arg(100)
->Arg(250)
->Arg(500)
->MinTime(2.0)
->Repetitions(5);
//BENCHMARK(BM_MultiplicationJIK)
//->Arg(10)
//->Arg(50)
//->Arg(100)
//->Arg(250)
//->Arg(500);
//BENCHMARK(BM_MultiplicationJKI)
//->Arg(10)
//->Arg(50)
//->Arg(100)
//->Arg(250)
//->Arg(500);
BENCHMARK(BM_MultiplicationKIJ)
->Arg(10)
->Arg(50)
->Arg(100)
->Arg(250)
->Arg(500)
->MinTime(2.0)
->Repetitions(5);
//BENCHMARK(BM_MultiplicationKJI)
//->Arg(10)
//->Arg(50)
//->Arg(100)
//->Arg(250)
//->Arg(500);

BENCHMARK_MAIN();

//int main(int argc, char** argv) {
//
//	benchmark::RegisterBenchmark(
//		"BM_IJK",
//		[](benchmark::State& state) {
//			BM_Multiplication(state, multiplyIJK);
//		}
//	);
//
//	benchmark::Initialize(&argc, argv);
//	benchmark::RunSpecifiedBenchmarks();
//	benchmark::Shutdown();
//
//	return 0;
//}