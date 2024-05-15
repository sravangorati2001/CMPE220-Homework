
#include "AccelerateUtils.h"

// Function to flatten a 2D matrix
std::vector<double> flatten_matrix(const std::vector<std::vector<double>> &mat) {
    std::vector<double> flat(mat.size() * mat[0].size());
    for (size_t i = 0; i < mat.size(); ++i) {
        std::copy(mat[i].begin(), mat[i].end(), flat.begin() + i * mat[0].size());
    }
    return flat;
}

std::vector<double> mat_vec_mult_accelerate(const std::vector<std::vector<double>> &mat, const std::vector<double> &vec) {
    size_t rows = mat.size();
    size_t cols = vec.size();
    std::vector<double> result(rows, 0.0);
    std::vector<double> flat_mat = flatten_matrix(mat);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, (int)rows, (int)cols, 1.0, flat_mat.data(), (int)cols, vec.data(), 1, 0.0, result.data(), 1);
    return result;
}

std::vector<double> vec_add_accelerate(const std::vector<double> &vec1, std::vector<double> &vec2) {
    size_t size = vec1.size();
    cblas_daxpy((int)size, 1.0, vec1.data(), 1, vec2.data(), 1);
    return vec2;
}

