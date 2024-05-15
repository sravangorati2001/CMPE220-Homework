#include <iostream>
#include <vector>
#include <chrono>
#include "BasicFFT.hpp"
#include "AccelerateFFT.hpp"
#include "MetalFFT.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

int main() {
    const size_t N = 50000; // Adjust size as needed
    std::vector<std::complex<double>> input_double(N, {1.0, 0.0}); // Example input for Basic FFT
    std::vector<std::complex<float>> input_float(N, {1.0f, 0.0f}); // Example input for Accelerate and Metal FFT

    // Basic FFT
    auto start = std::chrono::high_resolution_clock::now();
    auto result_basic = fft_basic(input_double);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_basic = end - start;
    std::cout << "Basic FFT Result Size: " << result_basic.size() << std::endl;
    std::cout << "Basic FFT Duration: " << duration_basic.count() << " seconds" << std::endl;

    // Accelerated FFT
    start = std::chrono::high_resolution_clock::now();
    auto result_accelerate = fft_accelerate(input_float);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_accelerate = end - start;
    std::cout << "Accelerate FFT Result Size: " << result_accelerate.size() << std::endl;
    std::cout << "Accelerate FFT Duration: " << duration_accelerate.count() << " seconds" << std::endl;

    @autoreleasepool {
        std::cout << "Starting Metal FFT computation..." << std::endl;

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
        NSString* filePath = [[NSBundle mainBundle] pathForResource:@"default" ofType:@"metallib"];
        id<MTLLibrary> library = [device newLibraryWithFile:filePath error:&error];
        if (!library || error) {
            std::cerr << "Failed to load Metal library: " << (error ? error.localizedDescription.UTF8String : "Unknown error") << std::endl;
            return -1;
        }
        std::cout << "Metal library loaded: " << library << std::endl;

        id<MTLFunction> fftFunction = [library newFunctionWithName:@"fft"];
        if (!fftFunction) {
            std::cerr << "Failed to load Metal FFT function." << std::endl;
            return -1;
        }
        std::cout << "FFT Function loaded: " << fftFunction << std::endl;

        id<MTLComputePipelineState> fftPipeline = [device newComputePipelineStateWithFunction:fftFunction error:&error];
        if (error) {
            std::cerr << "Failed to create FFT pipeline: " << error.localizedDescription.UTF8String << std::endl;
            return -1;
        }
        std::cout << "FFT Pipeline created successfully." << std::endl;

        // Sample Data for Metal
        std::vector<std::complex<float>> input_metal(N, {1.0f, 0.0f});
        std::vector<std::complex<float>> output_metal(N);

        // Create Buffers for Metal
        id<MTLBuffer> inputBuffer = createComplexBuffer(device, input_metal);
        id<MTLBuffer> outputBuffer = createComplexBuffer(device, output_metal);
        std::cout << "Buffers created." << std::endl;

        // Encode FFT
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> fftEncoder = [commandBuffer computeCommandEncoder];
        [fftEncoder setComputePipelineState:fftPipeline];
        [fftEncoder setBuffer:inputBuffer offset:0 atIndex:0];
        [fftEncoder setBuffer:outputBuffer offset:0 atIndex:1];
        
        MTLSize threadsPerThreadgroup = MTLSizeMake(256, 1, 1);
        MTLSize threadgroupsPerGrid = MTLSizeMake((N + 255) / 256, 1, 1);
        [fftEncoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [fftEncoder endEncoding];
        std::cout << "FFT encoded." << std::endl;

        // Execute Commands
        start = std::chrono::high_resolution_clock::now();
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration_metal = end - start;
        std::cout << "Commands executed." << std::endl;

        // Read Result
        std::complex<float>* resultPointer = reinterpret_cast<std::complex<float>*>([outputBuffer contents]);
        std::vector<std::complex<float>> finalResult(resultPointer, resultPointer + N);
        std::cout << "Results read from buffer." << std::endl;

        // Output Results
        std::cout << "Metal FFT Result Size: " << finalResult.size() << std::endl;
        std::cout << "Metal FFT Duration: " << duration_metal.count() << " seconds" << std::endl;
    }

    return 0;
}
