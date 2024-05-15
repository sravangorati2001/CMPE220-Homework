//
//  AccelerateFFT.hpp
//  CMPE200_FFT
//
//  Created by Sravan Kumar Gorati on 5/12/24.
//

#ifndef ACCELERATEFFT_H
#define ACCELERATEFFT_H

#include <vector>
#include <complex>
#include <Accelerate/Accelerate.h>

// Accelerated FFT Implementation using Accelerate
std::vector<std::complex<float>> fft_accelerate(const std::vector<std::complex<float>> &input);

#endif // ACCELERATEFFT_H





