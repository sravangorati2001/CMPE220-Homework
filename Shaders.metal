//
//  Shaders.metal
//  CMPE220_metal
//
//  Created by Sravan Kumar Gorati on 5/10/24.
//


#include <metal_stdlib>
using namespace metal;

kernel void matVecMult(device const float* mat [[buffer(0)]],
                       device const float* vec [[buffer(1)]],
                       device float* result [[buffer(2)]],
                       uint id [[thread_position_in_grid]]) {
    float sum = 0.0;
    for (int i = 0; i < 20000; ++i) {
        sum += mat[id * 20000 + i] * vec[i];
    }
    result[id] = sum;
}

kernel void vecAdd(device const float* vec1 [[buffer(0)]],
                   device float* vec2 [[buffer(1)]],
                   uint id [[thread_position_in_grid]]) {
    vec2[id] += vec1[id];
}
