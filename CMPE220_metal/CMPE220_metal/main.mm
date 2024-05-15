#include <iostream>
#include <vector>
#include <chrono>
#include "BasicUtils.h"
#include "AccelerateUtils.h"
#include "MetalUtils.h"
#import <Foundation/Foundation.h>

int main() {
    // Sample Data for Performance Testing
    std::vector<std::vector<double>> matrix(10000, std::vector<double>(10000, 1.0)); // Reduced size for demonstration
    std::vector<double> vector(10000, 1.0);
    std::vector<double> bias(10000, 1.0);

    // Basic Operations
    auto start = std::chrono::high_resolution_clock::now();
    auto result_basic = mat_vec_mult_basic(matrix, vector);
    result_basic = vec_add_basic(result_basic, bias);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_basic = end - start;
    std::cout << "Basic Result Size: " << result_basic.size() << std::endl;
    std::cout << "Basic Duration: " << duration_basic.count() << " seconds" << std::endl;

    // Accelerated Operations with Accelerate
    start = std::chrono::high_resolution_clock::now();
    auto result_accelerate = mat_vec_mult_accelerate(matrix, vector);
    result_accelerate = vec_add_accelerate(result_accelerate, bias);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_accelerate = end - start;
    std::cout << "Accelerate Result Size: " << result_accelerate.size() << std::endl;
    std::cout << "Accelerate Duration: " << duration_accelerate.count() << " seconds" << std::endl;

    @autoreleasepool {
        std::cout << "Starting Metal computation..." << std::endl;

        // Initialize Metal
        NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
        if ([devices count] == 0) {
            std::cerr << "No Metal devices are available." << std::endl;
            return -1;
        }
        id<MTLDevice> device = [devices objectAtIndex:0];
        std::cout << "Metal device created: " << device << std::endl;

        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        NSError* error = nil;
        id<MTLLibrary> library = [device newDefaultLibrary];
        if (!library) {
            std::cerr << "Failed to load Metal library." << std::endl;
            return -1;
        }
        std::cout << "Metal library loaded: " << library << std::endl;

        id<MTLFunction> matVecMultFunction = [library newFunctionWithName:@"matVecMult"];
        id<MTLFunction> vecAddFunction = [library newFunctionWithName:@"vecAdd"];
        if (!matVecMultFunction || !vecAddFunction) {
            std::cerr << "Failed to load Metal functions." << std::endl;
            return -1;
        }
        std::cout << "Functions loaded: matVecMultFunction = " << matVecMultFunction << ", vecAddFunction = " << vecAddFunction << std::endl;

        id<MTLComputePipelineState> matVecMultPipeline = [device newComputePipelineStateWithFunction:matVecMultFunction error:&error];
        if (error) {
            std::cerr << "Failed to create matVecMult pipeline: " << error.localizedDescription.UTF8String << std::endl;
            return -1;
        }
        id<MTLComputePipelineState> vecAddPipeline = [device newComputePipelineStateWithFunction:vecAddFunction error:&error];
        if (error) {
            std::cerr << "Failed to create vecAdd pipeline: " << error.localizedDescription.UTF8String << std::endl;
            return -1;
        }
        std::cout << "Pipelines created successfully." << std::endl;

        // Sample Data for Metal
        int rows = 10000;
        int cols = 10000;
        std::vector<float> matrix_flat(rows * cols, 1.0f);
        std::vector<float> vector_flat(cols, 1.0f);
        std::vector<float> bias_flat(rows, 1.0f);
        std::vector<float> result_flat(rows, 0.0f);
        std::cout << "Data initialized." << std::endl;

        // Create Buffers for Metal
        id<MTLBuffer> matBuffer = createBuffer(device, matrix_flat);
        id<MTLBuffer> vecBuffer = createBuffer(device, vector_flat);
        id<MTLBuffer> resultBuffer = createBuffer(device, result_flat);
        id<MTLBuffer> biasBuffer = createBuffer(device, bias_flat);
        std::cout << "Buffers created." << std::endl;

        // Encode Matrix-Vector Multiplication
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> matVecMultEncoder = [commandBuffer computeCommandEncoder];
        [matVecMultEncoder setComputePipelineState:matVecMultPipeline];
        [matVecMultEncoder setBuffer:matBuffer offset:0 atIndex:0];
        [matVecMultEncoder setBuffer:vecBuffer offset:0 atIndex:1];
        [matVecMultEncoder setBuffer:resultBuffer offset:0 atIndex:2];
        
        MTLSize threadsPerThreadgroup = MTLSizeMake(256, 1, 1);
        MTLSize threadgroupsPerGrid = MTLSizeMake((rows + 255) / 256, 1, 1);
        [matVecMultEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [matVecMultEncoder endEncoding];
        std::cout << "Matrix-vector multiplication encoded." << std::endl;

        // Encode Vector Addition
        id<MTLComputeCommandEncoder> vecAddEncoder = [commandBuffer computeCommandEncoder];
        [vecAddEncoder setComputePipelineState:vecAddPipeline];
        [vecAddEncoder setBuffer:resultBuffer offset:0 atIndex:0];
        [vecAddEncoder setBuffer:biasBuffer offset:0 atIndex:1];
        [vecAddEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [vecAddEncoder endEncoding];
        std::cout << "Vector addition encoded." << std::endl;

        // Execute Commands
        start = std::chrono::high_resolution_clock::now();
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration_metal = end - start;
        std::cout << "Commands executed." << std::endl;

        // Read Result
        float* resultPointer = reinterpret_cast<float*>([resultBuffer contents]);
        std::vector<float> finalResult(resultPointer, resultPointer + rows);
        std::cout << "Results read from buffer." << std::endl;

        // Output Results
        std::cout << "Metal Result Size: " << finalResult.size() << std::endl;
        std::cout << "Metal Duration: " << duration_metal.count() << " seconds" << std::endl;
        std::cout << "First 10 results: ";
        for (int i = 0; i < 10; ++i) {
            std::cout << finalResult[i] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
