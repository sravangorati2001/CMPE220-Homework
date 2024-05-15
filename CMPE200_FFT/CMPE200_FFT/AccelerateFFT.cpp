//
//  AccelerateFFT.cpp
//  CMPE200_FFT
//
//  Created by Sravan Kumar Gorati on 5/12/24.
//

#include "AccelerateFFT.hpp"

std::vector<std::complex<float>> fft_accelerate(const std::vector<std::complex<float>> &input) {
    size_t n = input.size();
    std::vector<std::complex<float>> output(n);

    // Create split complex format required by Accelerate
    DSPSplitComplex splitComplex;
    std::vector<float> real(n);
    std::vector<float> imag(n);
    splitComplex.realp = real.data();
    splitComplex.imagp = imag.data();

    // Convert input to split complex format
    for (size_t i = 0; i < n; ++i) {
        splitComplex.realp[i] = input[i].real();
        splitComplex.imagp[i] = input[i].imag();
    }

    // Setup FFT
    FFTSetup fftSetup = vDSP_create_fftsetup(log2(n), kFFTRadix2);

    // Perform FFT
    vDSP_fft_zip(fftSetup, &splitComplex, 1, log2(n), FFT_FORWARD);

    // Convert output back to std::complex
    for (size_t i = 0; i < n; ++i) {
        output[i] = std::complex<float>(splitComplex.realp[i], splitComplex.imagp[i]);
    }

    // Clean up
    vDSP_destroy_fftsetup(fftSetup);

    return output;
}







