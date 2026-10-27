#include <iostream>
#include <linalg/Matrix.hpp>
#include <stdexcept>
#include <iomanip>

int main()
{   
    double a = 1e16;
    double b = 1.0000000000000002;
    double c = -1e16;

    double ordinary = a * b + c;
    double fused = std::fma(a, b, c);

    std::cout << std::setprecision(17);

    std::cout << "Ordinary: " << ordinary << '\n';
    std::cout << "FMA:      " << fused << '\n';

    return 0;
}

