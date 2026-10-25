#include <iostream>
#include <linalg/Matrix.hpp>
#include <stdexcept>
#include <iomanip>

int main()
{   
    /*linalg::Matrix A(2, 3);

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

    linalg::Matrix C(10, 10);

    C(6, 1) = 3;
    C(8, 1) = 3;*/

    /*try {
        std::cout << transpose(transpose(A));
    }
    catch (const std::invalid_argument& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }*/
    
    /*try {
        std::cout << A;
        std::cout << A.combineRows(-1, 0, 1).scaleRow((1.0 / 3), 1);
    }
    catch (const std::out_of_range e) {
        std::cout << e.what();
    }
    catch (const std::invalid_argument e) {
        std::cout << e.what();
    }*/
    // std::cout << (A != B);

    /*std::cout << C;*/

    /*if (C.findFirstPivot(1, 0).has_value()) {
        std::cout << "There is a pivot in row: " << C.findFirstPivot(1).value() << std::endl;
    }
    else {
        std::cout << "There is no pivot in this column";
    }*/


    linalg::Matrix A(3, 3);

    A(0, 0) = 2;
    A(0, 1) = 4;
    A(0, 2) = 2;

    A(1, 0) = 4;
    A(1, 1) = 10;
    A(1, 2) = 6;

    A(2, 0) = 2;
    A(2, 1) = 8;
    A(2, 2) = 10;

    linalg::Matrix B(3, 3);

    B(0, 0) = 0;
    B(0, 1) = 2;
    B(0, 2) = 4;

    B(1, 0) = 2;
    B(1, 1) = 4;
    B(1, 2) = 6;

    B(2, 0) = 4;
    B(2, 1) = 6;
    B(2, 2) = 8;

    linalg::Matrix C(3, 4);

    C(0, 0) = 0;
    C(0, 1) = 2;
    C(0, 2) = 4;
    C(0, 3) = 6;

    C(1, 0) = 0;
    C(1, 1) = 4;
    C(1, 2) = 10;
    C(1, 3) = 14;

    C(2, 0) = 0;
    C(2, 1) = 6;
    C(2, 2) = 12;
    C(2, 3) = 18;


    try {
        std::cout << "Matrix A:\n\n" << A << "\n\n";
        std::cout << "A after gaussian reduction: \n\n" << A.toEchelonForm() << "\n\n";
    }
    catch (const std::invalid_argument e) {
        std::cout << "Error: " << e.what();
    }
    catch (const std::out_of_range e) {
        std::cout << "Error: " << e.what();
    }


    try {
        std::cout << "Matrix B:\n\n" << B << "\n\n";
        std::cout << "B after gaussian reduction: \n\n" << B.toEchelonForm() << "\n\n";
    }
    catch (const std::invalid_argument e) {
        std::cout << "Error: " << e.what();
    }
    catch (const std::out_of_range e) {
        std::cout << "Error: " << e.what();
    }


    try {
        std::cout << "Matrix C:\n\n" << C << "\n\n";
        std::cout << "C after gaussian reduction: \n\n" << C.toEchelonForm() << "\n\n";
    }
    catch (const std::invalid_argument e) {
        std::cout << "Error: " << e.what();
    }
    catch (const std::out_of_range e) {
        std::cout << "Error: " << e.what();
    }

    return 0;
}

