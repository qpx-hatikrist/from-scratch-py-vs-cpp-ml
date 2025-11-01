#include <iostream>
#include "otherFuncs.h"
#include "matrixFuncs.h"
#include "alias.h"

#define DEBUG true

int main() {
    std::vector<std::string> labels;
    matrix<float> mat;
    ReadCSV(labels, mat, "../../data/regression/advertising.csv");
    //ReadCSV(labels, mat, "../../data/regression/AMES_Final_DF.csv");
    mat = MatrixShuffleRow(mat);
    auto a = MatrixSplit(mat, 0.98f);

    #if DEBUG == true
    for (const auto& s : labels) printf("%10s\t", s.c_str());
    printf("\n");
    for (const auto& row : mat) {
        for (const auto& v : row) {
            printf("%f\t", v);
        }
        printf("\n");
    }

    for (size_t i = 0; i != a.size(); ++i) {
        printf("\nAfter split, matrix %ld:\n", i+1);
        for (const auto& row : a[i]) {
            for (const auto& v : row) {
                printf("%f\t", v);
            }
            printf("\n");
        }
    }

    std::cout << "Matrix 0/other size = " << mat.size() << ' ';
    for (const auto& m : a) std::cout << m.size() << ' ';
    std::cout << '\n';
    #endif

    std::cout << "TEST cut colum:\nMatrix after col extraction:\n";
    auto vec = MatrixExtractColumn(a.back(), 3);

    for (const auto& row : a.back()) {
        for (const auto& v : row) printf("%f\t", v);
        printf("\n");
    }
    std::cout << "Vector that was extracted:\n";
    for (const auto& row : vec) {
        for (const auto& v : row) printf("%f\t", v);
        printf("\n");
    }

    return 0;
}
