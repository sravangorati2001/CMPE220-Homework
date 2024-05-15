#include "MetalFFT.h"

id<MTLBuffer> createComplexBuffer(id<MTLDevice> device, const std::vector<std::complex<float>> &array) {
    return [device newBufferWithBytes:array.data()
                               length:array.size() * sizeof(std::complex<float>)
                              options:MTLResourceStorageModeShared];
}
