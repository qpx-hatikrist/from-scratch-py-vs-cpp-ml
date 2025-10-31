#include <iostream>
#include "otherFuncs.h"

#define DEBUG true

int main() {
    std::vector<std::string> labels;
    std::vector<std::vector<float>> data;
    ReadCSV(labels, data, "../../data/regression/advertising.csv");


    #if DEBUG == true
    for (const auto& w : labels) printf("%10s\t", w.c_str());
    printf("\n");
    for (const auto& row : data) {
        for (const auto& v : row) printf("%f\t", v);
        printf("\n");
    }
    #endif

    return 0;
}
