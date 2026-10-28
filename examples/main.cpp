#include <iostream>
#include <linalg/Matrix.hpp>
#include <linalg/PermutationMatrix.hpp>
#include <stdexcept>
#include <iomanip>

int main()
{   
    linalg::PermutationMatrix A(6);

    A.swapRows(0, 3);

    A.swapRows(3, 2);

    A.swapRows(0, 5);

    A.swapRows(1, 4);

    std::cout << A;

    return 0;
}

