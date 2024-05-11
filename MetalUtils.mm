#include "MetalUtils.h"
#include <iostream>

id<MTLBuffer> createBuffer(id<MTLDevice> device, const std::vector<float>& array) {
    return [device newBufferWithBytes:array.data()
                               length:array.size() * sizeof(float)
                              options:MTLResourceStorageModeShared];
}

void print_vector(const std::vector<float>& vec) {
    for (float val : vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}
