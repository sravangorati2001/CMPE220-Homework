//
//  BasicUtils.h
//  CMPE220_metal
//
//  Created by Sravan Kumar Gorati on 5/10/24.
//

#ifndef BASICUTILS_H
#define BASICUTILS_H

#include <vector>

// Function to print vector
void print_vector(const std::vector<double> &vec);

// Basic Matrix-Vector Multiplication
std::vector<double> mat_vec_mult_basic(const std::vector<std::vector<double>> &mat, const std::vector<double> &vec);

// Basic Vector Addition
std::vector<double> vec_add_basic(const std::vector<double> &vec1, const std::vector<double> &vec2);

#endif // BASICUTILS_H



