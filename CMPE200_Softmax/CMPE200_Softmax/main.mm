#include <iostream>
#include <vector>
#include <chrono>
#include "BasicSoftmax.hpp"
#include "AccelerateSoftmax.hpp"
#include "MetalSoftmax.h"
#include "BLASSoftmax.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

int main() {
    const size_t N = 50000; // Adjust size as needed
    std::vector<double> input_double(N, 1.0); // Example input for Basic, Accelerate, and BLAS Softmax
    std::vector<float> input_float(N, 1.0f);  // Example input for Metal Softmax
    std::vector<float> output(N);

    // Basic Softmax
    auto start = std::chrono::high_resolution_clock::now();
    auto result_basic = softmax_basic(input_double);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_basic = end - start;
    std::cout << "Basic Softmax Result Size: " << result_basic.size() << std::endl;
    std::cout << "Basic Softmax Duration: " << duration_basic.count() << " seconds" << std::endl;

    // Accelerated Softmax
    start = std::chrono::high_resolution_clock::now();
    auto result_accelerate = softmax_accelerate(input_double);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_accelerate = end - start;
    std::cout << "Accelerate Softmax Result Size: " << result_accelerate.size() << std::endl;
    std::cout << "Accelerate Softmax Duration: " << duration_accelerate.count() << " seconds" << std::endl;

    // BLAS Softmax
//    start = std::chrono::high_resolution_clock::now();
//    auto result_blas = softmax_blas(input_double);
//    end = std::chrono::high_resolution_clock::now();
//    std::chrono::duration<double> duration_blas = end - start;
//    std::cout << "BLAS Softmax Result Size: " << result_blas.size() << std::endl;
//    std::cout << "BLAS Softmax Duration: " << duration_blas.count() << " seconds" << std::endl;

    // Metal Softmax
    start = std::chrono::high_resolution_clock::now();
    softmax_metal(input_float, output);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_metal = end - start;
    std::cout << "Metal Softmax Result Size: " << output.size() << std::endl;
    std::cout << "Metal Softmax Duration: " << duration_metal.count() << " seconds" << std::endl;

    return 0;
}
