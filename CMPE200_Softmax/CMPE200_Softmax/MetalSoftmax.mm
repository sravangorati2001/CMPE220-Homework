#include "MetalSoftmax.h"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

id<MTLBuffer> createBuffer(id<MTLDevice> device, const std::vector<float> &array) {
    return [device newBufferWithBytes:array.data()
                               length:array.size() * sizeof(float)
                              options:MTLResourceStorageModeShared];
}

id<MTLBuffer> createScalarBuffer(id<MTLDevice> device, uint value) {
    return [device newBufferWithBytes:&value
                               length:sizeof(uint)
                              options:MTLResourceStorageModeShared];
}

void softmax_metal(const std::vector<float> &input, std::vector<float> &output) {
    @autoreleasepool {
        NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
        if ([devices count] == 0) {
            NSLog(@"No Metal devices are available.");
            return;
        }
        id<MTLDevice> device = [devices objectAtIndex:0];
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        NSError *error = nil;
        NSString *filePath = [[NSBundle mainBundle] pathForResource:@"default" ofType:@"metallib"];
        if (!filePath) {
            NSLog(@"Failed to find Metal library file path.");
            return;
        }

        id<MTLLibrary> library = [device newLibraryWithFile:filePath error:&error];
        if (!library || error) {
            NSLog(@"Failed to load Metal library: %@", error ? error.localizedDescription : @"Unknown error");
            return;
        }

        id<MTLFunction> softmaxFunction = [library newFunctionWithName:@"softmax"];
        if (!softmaxFunction) {
            NSLog(@"Failed to load Metal Softmax function.");
            return;
        }

        id<MTLComputePipelineState> softmaxPipeline = [device newComputePipelineStateWithFunction:softmaxFunction error:&error];
        if (error) {
            NSLog(@"Failed to create Softmax pipeline: %@", error.localizedDescription);
            return;
        }

        std::vector<float> exp_values(input.size());
        id<MTLBuffer> inputBuffer = createBuffer(device, input);
        id<MTLBuffer> outputBuffer = createBuffer(device, exp_values);

        uint gridSize = static_cast<uint>(input.size());
        id<MTLBuffer> gridSizeBuffer = createScalarBuffer(device, gridSize);

        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:softmaxPipeline];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:outputBuffer offset:0 atIndex:1];
        [encoder setBuffer:gridSizeBuffer offset:0 atIndex:2];

        MTLSize threadsPerThreadgroup = MTLSizeMake(256, 1, 1);
        MTLSize threadgroupsPerGrid = MTLSizeMake((input.size() + 255) / 256, 1, 1);
        [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        float *resultPointer = (float *)[outputBuffer contents];
        output.assign(resultPointer, resultPointer + input.size());
    }
}
