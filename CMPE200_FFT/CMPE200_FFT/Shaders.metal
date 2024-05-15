//
//  Shaders.metal
//  CMPE200_FFT
//
//  Created by Sravan Kumar Gorati on 5/12/24.
//

#include <metal_stdlib>
using namespace metal;

kernel void fft(
    constant float2 *input [[buffer(0)]],
    device float2 *output [[buffer(1)]],
    uint id [[thread_position_in_grid]]) {
    // FFT implementation (this is a placeholder, implement FFT logic here)
    output[id] = input[id];
}



