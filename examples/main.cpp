#include <iostream>
#include <linalg/Matrix.hpp>

int main()
{
    linalg::Matrix A(2, 3);
    
	A(5, 1) = 7.0; // This line will cause a runtime error because it tries to access an out-of-bounds index
    /*A(0, 1) = 7.0;
    A(1, 2) = 5.0;*/

    //std::cout << A(0, 1) << std::endl;
    return 0;
}

