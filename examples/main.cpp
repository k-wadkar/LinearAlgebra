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

    /*try {
        std::cout << transpose(transpose(A));
    }
    catch (const std::invalid_argument& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }*/
    
    try {
        std::cout << A;
        std::cout << A.combineRows(-1, 0, 1).scaleRow((1.0 / 3), 1);
    }
    catch (const std::out_of_range e) {
        std::cout << e.what();
    }
    catch (const std::invalid_argument e) {
        std::cout << e.what();
    }
    // std::cout << (A != B);

    return 0;
}

