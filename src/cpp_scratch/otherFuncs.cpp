#include "otherFuncs.h"

// Split line into vector of strings on delimiter
std::vector<std::string> SplitLineDelimiter(const std::string& line, char delim) {
    std::vector<std::string> vec;
    size_t pos = 0, i = 0;
    for (; i != line.size(); ++i) {
        if (line[i] == delim) {
            vec.push_back(line.substr(pos, i-pos));
            pos = i+1;
        }
    }
    vec.push_back(line.substr(pos, i-pos));
    return vec;
}
