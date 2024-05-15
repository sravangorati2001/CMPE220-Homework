//
//  BasicFFT.hpp
//  CMPE200_FFT
//
//  Created by Sravan Kumar Gorati on 5/12/24.
//

#ifndef BASICFFT_H
#define BASICFFT_H

#include <vector>
#include <complex>

// Function to print complex vector
void print_complex_vector(const std::vector<std::complex<double>> &vec);

// Basic FFT Implementation
std::vector<std::complex<double>> fft_basic(const std::vector<std::complex<double>> &input);

#endif // BASICFFT_H






