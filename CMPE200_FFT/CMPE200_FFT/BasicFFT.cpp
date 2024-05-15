//
//  BasicFFT.cpp
//  CMPE200_FFT
//
//  Created by Sravan Kumar Gorati on 5/12/24.
//

#include "BasicFFT.hpp"
#include <iostream>
#include <cmath>

// Function to print complex vector
void print_complex_vector(const std::vector<std::complex<double>> &vec) {
    for (const auto &val : vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

// Basic FFT Implementation
std::vector<std::complex<double>> fft_basic(const std::vector<std::complex<double>> &input) {
    size_t n = input.size();
    if (n <= 1) return input;

    std::vector<std::complex<double>> even(n / 2);
    std::vector<std::complex<double>> odd(n / 2);
    for (size_t i = 0; i < n / 2; ++i) {
        even[i] = input[i * 2];
        odd[i] = input[i * 2 + 1];
    }

    even = fft_basic(even);
    odd = fft_basic(odd);

    std::vector<std::complex<double>> result(n);
    for (size_t k = 0; k < n / 2; ++k) {
        std::complex<double> t = std::polar(1.0, -2 * M_PI * k / n) * odd[k];
        result[k] = even[k] + t;
        result[k + n / 2] = even[k] - t;
    }
    return result;
}





