//
// Created by William Zhao on 11/02/23.
//

#ifndef CRACK_DES_CSV_H
#define CRACK_DES_CSV_H

#include <iostream>
#include <vector>

class CSV {
private:
    std::string filename;
public:
    CSV(std::string);
    void customWrite(const std::string h, std::vector<int> t, const int size);
    void write(const std::vector<int> &sTimes,
               const std::vector<int> &pTimes,
               const std::vector<float> &spTimes,
               const int size);
};


#endif //CRACK_DES_CSV_H
