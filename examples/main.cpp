#include <iostream>
#include <linalg/Matrix.hpp>
#include <stdexcept>
#include <iomanip>

int main()
{   
    linalg::Matrix A(2, 3);

    A(0, 0) = 1;
    A(0, 1) = 2;
    A(0, 2) = 3;
    A(1, 0) = 4;
    A(1, 1) = 5;
    A(1, 2) = 6;

    linalg::Matrix B(4, 4);

    B(0, 0) = 3;
    B(1, 1) = 3;
    B(2, 2) = 3;

    try {
        std::cout << A * B;
    }
    catch (const std::invalid_argument& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }
    
    return 0;
}

