//
//  MetalFFT.h
//  CMPE200_FFT
//
//  Created by Sravan Kumar Gorati on 5/12/24.
//

#ifndef METALFFT_H
#define METALFFT_H

#include <vector>
#include <complex>
#import <Metal/Metal.h>

// Function to create Metal buffer for complex data
id<MTLBuffer> createComplexBuffer(id<MTLDevice> device, const std::vector<std::complex<float>> &array);

#endif // METALFFT_H





