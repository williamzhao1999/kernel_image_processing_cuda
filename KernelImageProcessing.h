//
// Created by pc23wilzha on 09/03/23.
//

#ifndef KERNEL_IMAGE_PROCESSING_SEQUENTIAL_KERNELIMAGEPROCESSING_H
#define KERNEL_IMAGE_PROCESSING_SEQUENTIAL_KERNELIMAGEPROCESSING_H

#include <iostream>
#include <vector>
#include <assert.h>
#include <cmath>
#include <png++/png.hpp>

using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image; //vector<vector<vector<double>>>

class KernelImageProcessing {

public:
    Matrix getSharpenKernelMatrix();
    Matrix getblurKernelMatrix();
    Matrix getGaussianBlurMatrix();
    float* getGaussian(int height, int width, double sigma);
    Image loadImage(const char *filename);
    float* loadImageFloatType(const char* filename);
    void saveImage(Image &image, const char *filename);
    Image convolute(Image& image, Matrix& filter, bool samePadding = false);
};


#endif //KERNEL_IMAGE_PROCESSING_SEQUENTIAL_KERNELIMAGEPROCESSING_H
