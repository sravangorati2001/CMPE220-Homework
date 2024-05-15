#include <metal_stdlib>
using namespace metal;

kernel void softmax(
    constant float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &gridSize [[buffer(2)]],
    uint id [[thread_position_in_grid]],
    uint threadGroupID [[thread_index_in_threadgroup]],
    uint numThreads [[threads_per_threadgroup]]) {

    if (id >= gridSize) return;

    // Shared memory for reduction
    threadgroup float shared_exp[256];
    threadgroup float max_val;

    // Step 1: Find the maximum value
    float local_max = -FLT_MAX;
    for (uint i = id; i < gridSize; i += numThreads) {
        local_max = max(local_max, input[i]);
    }

    // Reduce to find the global max
    if (threadGroupID < 256) {
        shared_exp[threadGroupID] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (threadGroupID == 0) {
        max_val = -FLT_MAX;
        for (uint i = 0; i < 256; ++i) {
            max_val = max(max_val, shared_exp[i]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Compute exponentials and sum them
    float sum_exp = 0.0;
    for (uint i = id; i < gridSize; i += numThreads) {
        output[i] = exp(input[i] - max_val);
        sum_exp += output[i];
    }

    // Reduce to find the sum of exponentials
    if (threadGroupID < 256) {
        shared_exp[threadGroupID] = sum_exp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (threadGroupID == 0) {
        sum_exp = 0.0;
        for (uint i = 0; i < 256; ++i) {
            sum_exp += shared_exp[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Normalize to get softmax probabilities
    for (uint i = id; i < gridSize; i += numThreads) {
        output[i] /= sum_exp;
    }
}
