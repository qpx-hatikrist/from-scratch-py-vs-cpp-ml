#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>


// Split line into vector of strings on delimiter
std::vector<std::string> SplitLineDelimiter(const std::string& line, char delim);

// Automatically convert string to desired type
template<typename T> 
T GetAs(const std::string& s) {
    std::stringstream ss(s);
    T t;
    ss >> t;
    return t;
}


// Reads CSV data into empty labels and data vectors
template<typename T>
void ReadCSV(std::vector<std::string>& labels, std::vector<std::vector<T>>& data, std::string fpath) {
    std::ifstream file(fpath);
    std::string line;
    std::getline(file, line);
    labels = SplitLineDelimiter(line, ',');
    while (std::getline(file, line)) {
        std::vector<std::string> temp = SplitLineDelimiter(line, ',');
        std::vector<T> dataRow;
        for (const auto& s : temp) dataRow.push_back(GetAs<T>(s));
        data.push_back(dataRow);
    }
}
