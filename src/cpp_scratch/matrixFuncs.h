#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include "alias.h"

// Shuffle matrix rows
template<typename T>
inline matrix<T> MatrixShuffleRow(matrix<T> mat) {
    std::shuffle(mat.begin(), mat.end(), std::default_random_engine(
        std::chrono::system_clock::now().time_since_epoch().count()
    ));
    return mat;
}

// Shuffle matrix rows with specified seed
template<typename T>
inline matrix<T> MatrixShuffleRow(matrix<T> mat, unsigned long seed) {
    std::shuffle(mat.begin(), mat.end(), std::default_random_engine(seed));
    return mat;
}

// Split matrix into two with ratio
template<typename T>
std::vector<matrix<T>> MatrixSplit(const matrix<T>& mat, float ratio) {
    size_t size = mat.size()*ratio;
    std::vector<matrix<T>> ret;
    if (ratio < 0.0f || ratio > 1.0f) return ret;
    ret.push_back(matrix<T>(mat.begin(), mat.begin()+size));
    ret.push_back(matrix<T>(mat.begin()+size, mat.end()));
    return ret;
}

// Split matrix into N matrices of same size
template<typename T>
std::vector<matrix<T>> MatrixSplit(const matrix<T>& mat, int N) {
    size_t size = mat.size()/N, extraSize = mat.size()-N*size;
    std::vector<matrix<T>> ret;
    if (N <= 0 || N >= mat.size()) return ret;
    for (int i = 0; i != N-1; ++i) {
        int step = i*size;
        ret.push_back(matrix<T>(mat.begin()+step, mat.begin()+step+size));
    }
    int step = (N-1)*size;
    ret.push_back(matrix<T>(mat.begin()+step, mat.begin()+step+size+extraSize));
    return ret;
}

// Extract target column and modify origin matrix
template<typename T>
matrix<T> MatrixExtractColumn(matrix<T>& mat, int targetCol) {
    matrix<T> ret = {};
    if (targetCol < 0 || size_t(targetCol) >= mat[0].size()) return ret;
    for (auto& row : mat) {
        ret.push_back(std::vector<T>{row[targetCol]});
        row.erase(row.begin()+targetCol);
    }
    return ret;
}




//////////////////////////////// MATH SECTION ////////////////////////////////
// All matrix math functions assume that input matrices are valid
// Therefore there is no error handling - beware of crashes!

// Calculate average value of matrix
template<typename T>
inline double CalculateAverage(const matrix<T>& m1) {
    double ans = 0.0d;
    int n = m1.size()*m1[0].size();
    for (size_t row = 0; row != m1.size(); ++row) {
        for (size_t col = 0; col != m1[0].size(); ++col) {
            ans += m1[row][col];
        }
    }
    return ans/n;
}

// Calculate mean absolute error
template<typename T>
double CalculateMAE(const matrix<T>& m1, const matrix<T>& m2) {
    double ans = 0.0d;
    int n = m1.size()*m1[0].size();
    for (size_t row = 0; row != m1.size(); ++row) {
        for (size_t col = 0; col != m1[0].size(); ++col) {
            ans += std::abs(m1[row][col] - m2[row][col]);
        }
    }
    return ans/n;
}

// Calculate mean absolute percentage error
template<typename T>
double CalculateMAPE(const matrix<T>& realMat, const matrix<T>& predictMat) {
    double ans = 0.0d;
    int n = realMat.size()*realMat[0].size();
    for (size_t row = 0; row != realMat.size(); ++row) {
        for (size_t col = 0; col != realMat[0].size(); ++col) {
            ans += (std::abs(realMat[row][col] - predictMat[row][col]))/realMat[row][col];
        }
    }
    return 100.0d*ans/n;
}
// Calculate mean square error
template<typename T>
double CalculateMSE(const matrix<T>& m1, const matrix<T>& m2) {
    double ans = 0.0d;
    int n = m1.size()*m1[0].size();
    for (size_t row = 0; row != m1.size(); ++row) {
        for (size_t col = 0; col != m1[0].size(); ++col) {
            T val = m1[row][col] - m2[row][col];
            ans += val*val;
        }
    }
    return ans/n;
}

// Calculate R squared (R2)
template<typename T>
double CalculateR2(const matrix<T>& realMat, const matrix<T>& predictMat) {
    double matAvg = CalculateAverage(realMat);
    double squareAvg = 0.0d, squarePred = 0.0d;
    for (size_t row = 0; row != realMat.size(); ++row) {
        for (size_t col = 0; col != realMat[0].size(); ++col) {
            double tempAvg  = realMat[row][col] - matAvg,
                   tempPred = realMat[row][col] - predictMat[row][col];
            squareAvg  += tempAvg*tempAvg;
            squarePred += tempPred*tempPred;
        }
    }
    return 1.0d - squarePred/squareAvg;
}


