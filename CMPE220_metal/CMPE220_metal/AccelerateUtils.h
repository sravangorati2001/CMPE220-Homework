//
//  AccelerateUtils.h
//  CMPE220_metal
//
//  Created by Sravan Kumar Gorati on 5/10/24.
//

#ifndef ACCELERATEUTILS_H
#define ACCELERATEUTILS_H

#include <vector>
#include <Accelerate/Accelerate.h>

// Accelerated Matrix-Vector Multiplication with Accelerate
std::vector<double> mat_vec_mult_accelerate(const std::vector<std::vector<double>> &mat, const std::vector<double> &vec);

// Accelerated Vector Addition with Accelerate
std::vector<double> vec_add_accelerate(const std::vector<double> &vec1, std::vector<double> &vec2);

#endif // ACCELERATEUTILS_H

