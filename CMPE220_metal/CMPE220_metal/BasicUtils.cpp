//
//  BasicUtils.cpp
//  CMPE220_metal
//
//  Created by Sravan Kumar Gorati on 5/10/24.
//

#include "BasicUtils.h"
#include <iostream>

void print_vector(const std::vector<double> &vec) {
    for (double val : vec) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

std::vector<double> mat_vec_mult_basic(const std::vector<std::vector<double>> &mat, const std::vector<double> &vec) {
    size_t rows = mat.size();
    size_t cols = vec.size();
    std::vector<double> result(rows, 0.0);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i] += mat[i][j] * vec[j];
        }
    }
    return result;
}

std::vector<double> vec_add_basic(const std::vector<double> &vec1, const std::vector<double> &vec2) {
    size_t size = vec1.size();
    std::vector<double> result(size, 0.0);
    for (size_t i = 0; i < size; ++i) {
        result[i] = vec1[i] + vec2[i];
    }
    return result;
}


