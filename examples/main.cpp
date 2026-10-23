#include <iostream>
#include <linalg/Matrix.hpp>
#include <stdexcept>

int main()
{
    const linalg::Matrix A(2, 3);
    
    try {
        std::cout << "Before access\n";
        std::cout << A(5, 1) << "\n"; // This line will cause a runtime error because it tries to access an out-of-bounds index
        std::cout << "After access\n";
    }
    catch (const std::out_of_range& e) {
        std::cout << "Error: " << e.what() << "\n";
    }
    
    std::cout << "Program finished" << std::endl;

    return 0;
}

