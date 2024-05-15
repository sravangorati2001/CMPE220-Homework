//
//  AccelerateSoftmax.cpp
//  CMPE200_Softmax
//
//  Created by Sravan Kumar Gorati on 5/12/24.
//

#include "AccelerateSoftmax.hpp"
#include <Accelerate/Accelerate.h>
#include <vector>

std::vector<double> softmax_accelerate(const std::vector<double> &input) {
    size_t n = input.size();
    std::vector<double> output(n);
    double max_val;
    vDSP_maxvD(input.data(), 1, &max_val, n);

    std::vector<double> exp_values(n);
    double sum_exp = 0.0;
    double neg_max_val = -max_val;

    vDSP_vsaddD(input.data(), 1, &neg_max_val, exp_values.data(), 1, n);

    // Convert size_t to int for vvsExp
    int n_int = static_cast<int>(n);
    vvexp(exp_values.data(), exp_values.data(), &n_int);

    vDSP_sveD(exp_values.data(), 1, &sum_exp, n);
    vDSP_vsdivD(exp_values.data(), 1, &sum_exp, output.data(), 1, n);

    return output;
}


