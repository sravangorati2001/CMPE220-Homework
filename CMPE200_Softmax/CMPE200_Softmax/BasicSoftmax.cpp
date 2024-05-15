//
//  BasicSoftmax.cpp
//  CMPE200_Softmax
//
//  Created by Sravan Kumar Gorati on 5/12/24.
//

#include "BasicSoftmax.hpp"
#include <cmath>
#include <vector>

std::vector<double> softmax_basic(const std::vector<double> &input) {
    std::vector<double> output(input.size());
    double max_val = *std::max_element(input.begin(), input.end());
    double sum = 0.0;

    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }

    for (size_t i = 0; i < input.size(); ++i) {
        output[i] /= sum;
    }

    return output;
}



