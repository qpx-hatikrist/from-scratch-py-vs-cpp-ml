#include <iostream>
#include "otherFuncs.h"
#include "matrixFuncs.h"
#include "mlAlgorithms.h"
#include "alias.h"

#define DEBUG true

int main() {
    std::vector<std::string> labels;
    matrix<float> mat;
    ReadCSV(labels, mat, "../../data/regression/advertising.csv");
    //ReadCSV(labels, mat, "../../data/regression/AMES_Final_DF.csv");
    mat = MatrixShuffleRow(mat);
    auto a = MatrixSplit(mat, 0.5f);
    auto vec = MatrixExtractColumn(a.back(), 3);
    return 0;
}
