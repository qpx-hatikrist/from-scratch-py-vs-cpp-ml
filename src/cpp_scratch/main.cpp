#include <iostream>
#include "otherFuncs.h"
#include "matrixFuncs.h"
#include "alias.h"

#define DEBUG true

int main() {
    std::vector<std::string> labels;
    myMatrix<float> mat;
    ReadCSV(labels, mat, "../../data/regression/advertising.csv");
    //ReadCSV(labels, mat, "../../data/regression/AMES_Final_DF.csv");
    mat = MatrixShuffleRow(mat);

    auto a = MatrixSplit(mat, 0.5f);

    #if DEBUG == true
    printf("Matrix def:\n");
    for (const auto& row : mat) {
        for (const auto& v : row) {
            printf("%f\t", v);
        }
        printf("\n");
    }

    for (int i = 0; i != a.size(); ++i) {
        printf("\nAfter split, matrix %d:\n", i+1);
        for (const auto& row : a[0]) {
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

    return 0;
}
