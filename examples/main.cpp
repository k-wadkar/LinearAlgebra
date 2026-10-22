#include <iostream>
#include <linalg/Matrix.hpp>

int main()
{
    linalg::Matrix A(2, 3);
    
    A(0, 1) = 7.0;
    A(1, 2) = 5.0;

    std::cout << "El1 " << A(0, 1) << ", El2: " << A(1, 2) << std::endl;
    return 0;
}

