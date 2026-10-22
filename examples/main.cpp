#include <iostream>
#include <linalg/Matrix.hpp>

int main()
{
    std::size_t r;
    std::size_t c;
    
	std::cout << "Enter number of rows: ";
	std::cin >> r;
    std::cout << "Enter number of columns: ";
    std::cin >> c;

    const linalg::Matrix A(r, c);
    
    std::cout << "Rows: " << A.rows() << ", Cols: " << A.cols() << std::endl;
    return 0;
}