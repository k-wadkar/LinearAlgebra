#include <iostream>
#include <linalg/Matrix.hpp>
#include <stdexcept>
#include <iomanip>

int main()
{
    //const linalg::Matrix A(2, 3);
    //
    //try {
    //    std::cout << "Before access\n";
    //    std::cout << A(5, 1) << "\n"; // This line will cause a runtime error because it tries to access an out-of-bounds index
    //    std::cout << "After access\n";
    //}
    //catch (const std::out_of_range& e) {
    //    std::cout << "Error: " << e.what() << "\n";
    //}
    //
    //std::cout << "Program finished" << std::endl;
    
    linalg::Matrix A(2, 3);

    A(0, 0) = 1;
    A(0, 1) = 2;
    A(0, 2) = 3;
    A(1, 0) = 4;
    A(1, 1) = 5;
    A(1, 2) = 6;
    
    std::cout << A;
    
    return 0;
}

