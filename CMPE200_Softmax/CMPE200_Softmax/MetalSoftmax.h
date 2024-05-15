//
//  MetalSoftmax.h
//  CMPE200_Softmax
//
//  Created by Sravan Kumar Gorati on 5/12/24.
//

#ifndef METALSOFTMAX_H
#define METALSOFTMAX_H

#include <vector>
#import <Metal/Metal.h>

// Function to create Metal buffer
id<MTLBuffer> createBuffer(id<MTLDevice> device, const std::vector<float> &array);
id<MTLBuffer> createScalarBuffer(id<MTLDevice> device, uint value);

// Metal Softmax Implementation
void softmax_metal(const std::vector<float> &input, std::vector<float> &output);

#endif // METALSOFTMAX_H


