#include <iostream>
#include <linalg/Matrix.hpp>
#include <stdexcept>
#include <iomanip>

int main()
{   
	linalg::Matrix B(3, 4);

	B(0, 0) = 0;
	B(0, 1) = 2;
	B(0, 2) = 4;
	B(0, 3) = 6;

	B(1, 0) = 0;
	B(1, 1) = 4;
	B(1, 2) = 10;
	B(1, 3) = 14;

	B(2, 0) = 0;
	B(2, 1) = 6;
	B(2, 2) = 12;
	B(2, 3) = 18;

	std::cout << "Matrix B:\n\n" << B << "\n\n\n";
	std::cout << "Matrix B after Gauss-Jordan reduction:\n\n" << B.toReducedEchelonForm();

    return 0;
}

