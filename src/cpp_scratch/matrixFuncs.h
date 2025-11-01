#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include "alias.h"

// Shuffle matrix rows
template<typename T>
inline myMatrix<T> MatrixShuffleRow(myMatrix<T> mat) {
    std::shuffle(mat.begin(), mat.end(), std::default_random_engine(
        std::chrono::system_clock::now().time_since_epoch().count()
    ));
    return mat;
}

// Split matrix into two with ratio
template<typename T>
std::vector<myMatrix<T>> MatrixSplit(const myMatrix<T>& mat, float ratio) {
    size_t size = mat.size()*ratio;
    std::vector<myMatrix<T>> ret;
    if (ratio < 0.0f || ratio > 1.0f) return ret;
    ret.push_back(myMatrix<T>(mat.begin(), mat.begin()+size));
    ret.push_back(myMatrix<T>(mat.begin()+size, mat.end()));
    return ret;
}

// Split matrix into N matrices of same size
template<typename T>
std::vector<myMatrix<T>> MatrixSplit(const myMatrix<T>& mat, int N) {
    size_t size = mat.size()/N, extraSize = mat.size()-N*size;
    std::vector<myMatrix<T>> ret;
    if (N <= 0 || N >= mat.size()) return ret;
    for (int i = 0; i != N-1; ++i) {
        int step = i*size;
        ret.push_back(myMatrix<T>(mat.begin()+step, mat.begin()+step+size));
    }
    int step = (N-1)*size;
    ret.push_back(myMatrix<T>(mat.begin()+step, mat.begin()+step+size+extraSize));
    return ret;
}
