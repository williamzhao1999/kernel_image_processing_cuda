//
// Created by William Zhao on 11/02/23.
//

#include "CSV.h"

#include <fstream>
#include <vector>
#include <iostream>
#include <string>

CSV::CSV(std::string f) : filename(f) {

}

void CSV::customWrite(const std::string h, std::vector<int> t, const int size){
    std::fstream fout;

    // opens an existing csv file or creates a new file.
    fout.open(filename, std::ios::out);

    fout << h << "\n";

    const int sizeVector = t.size();
    for(int i = 0; i < sizeVector; i++){
        fout << std::to_string(i+1) + "," + std::to_string(t[i]) + "\n";
    }

    fout.close();

}

void CSV::write(const std::vector<int> &sTimes,
                const std::vector<int> &pTimes,
                const std::vector<float> &spTimes,
                const int size) {
    std::fstream fout;

    // opens an existing csv file or creates a new file.
    fout.open(filename, std::ios::out);

    fout << "Length,Sequential Time,Parallel Time,Speed UP\n";

    for(int i = 0; i < size; i++){
        std::string formattedText = std::to_string(i+1) + "," +
                                    std::to_string(sTimes.at(i)) + "," + std::to_string(pTimes.at(i)) + "," +
                                    std::to_string(spTimes.at(i));
        fout << formattedText << "\n";
    }

    fout.close();


}