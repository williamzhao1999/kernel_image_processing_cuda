//
// Created by pc23wilzha on 09/03/23.
//

#include "KernelImageProcessing.h"

float* KernelImageProcessing::getGaussian(int height, int width, double sigma)
{
    float* kernel = new float[height*width];
    double sum=0.0;
    int i,j;

    for (i=0 ; i<height ; i++) {
        for (j=0 ; j<width ; j++) {
            kernel[i*height+width] = exp(-(i*i+j*j)/(2*sigma*sigma))/(2*M_PI*sigma*sigma);
            sum += kernel[i*height+width];
        }
    }

    for (i=0 ; i<height ; i++) {
        for (j=0 ; j<width ; j++) {
            kernel[i*height+width] /= sum;
        }
    }

    return kernel;
}

Matrix KernelImageProcessing::getSharpenKernelMatrix(){
    Matrix blur(3,Array(3));
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            blur[i][j] = 0;
        }
    }

    blur[0][1] = -1;
    blur[1][0] = -1;
    blur[1][1] = 5;
    blur[1][2] = -1;
    blur[2][1] = -1;

    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            std::cout << blur[i][j] << " ";
        }
        std::cout << std::endl;
    }
    return blur;
}

Matrix KernelImageProcessing::getblurKernelMatrix(){
    Matrix blur(3,Array(3));
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            blur[i][j] = 1;
            blur[i][j] /= 9;
            std::cout << blur[i][j] << " ";
        }
        std::cout << std::endl;
    }
    return blur;
}

Matrix KernelImageProcessing::getGaussianBlurMatrix(){
    Matrix blur(3,Array(3));

    blur[0][0] = 1;
    blur[0][1] = 2;
    blur[0][2] = 1;
    blur[1][0] = 2;
    blur[1][1] = 4;
    blur[1][2] = 2;
    blur[2][0] = 1;
    blur[2][1] = 2;
    blur[2][2] = 1;

    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            blur[i][j] /= 16;
            std::cout << blur[i][j] << " ";
        }
        std::cout << std::endl;
    }
    return blur;
}

Image KernelImageProcessing::loadImage(const char *filename)
{
    png::image<png::rgb_pixel> image(filename);
    Image imageMatrix(3, Matrix(image.get_height(), Array(image.get_width())));

    int h,w;
    for (h=0 ; h<image.get_height() ; h++) {
        for (w=0 ; w<image.get_width() ; w++) {
            imageMatrix[0][h][w] = image[h][w].red;
            imageMatrix[1][h][w] = image[h][w].green;
            imageMatrix[2][h][w] = image[h][w].blue;
        }
    }

    return imageMatrix;
}

float* KernelImageProcessing::loadImageFloatType(const char* filename)
{
    png::image<png::rgb_pixel> image(filename);

    int width = image.get_width();
    int height = image.get_height();

    int num_pixels = width * height;
    int float_array_size = num_pixels * 3;

    float* ptr = new float[float_array_size];

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            png::rgb_pixel pixel = image[y][x];

            *ptr++ = (float)pixel.red;
            *ptr++ = (float)pixel.green;
            *ptr++ = (float)pixel.blue;
        }
    }
}

void KernelImageProcessing::saveImage(Image &image, const char *filename)
{
    assert(image.size()==3);

    int height = image[0].size();
    int width = image[0][0].size();
    int x,y;

    png::image<png::rgb_pixel> imageFile(width, height);

    for (y=0 ; y<height ; y++) {
        for (x=0 ; x<width ; x++) {
            imageFile[y][x].red = image[0][y][x];
            imageFile[y][x].green = image[1][y][x];
            imageFile[y][x].blue = image[2][y][x];
        }
    }
    imageFile.write(filename);
}

Image KernelImageProcessing::convolute(Image& image, Matrix& filter, bool samePadding){

    int height = image[0].size();
    int width = image[0][0].size();
    int filterHeight = filter.size();
    int filterWidth = filter[0].size();

    int output_height = height - filterHeight + 1;
    int output_width = width - filterWidth + 1;

    int padHeight = filterHeight / 2;
    int padWidth = filterWidth / 2;
    Image paddedImage(3, Matrix(height+2*padHeight, Array(width+2*padWidth)));

    if(samePadding){


        for (int d=0 ; d<3 ; d++) {
            for (int i=padHeight ; i<height+padHeight ; i++) {
                for (int j=padWidth ; j<width+padWidth ; j++) {

                    paddedImage[d][i][j] = image[d][i-padHeight][j-padWidth];
                }
            }
        }

        output_height = height;
        output_width = width;
    }

    std::cout << "Captured Height Image: " << height << std::endl;
    std::cout << "Captured Width Image: " << width << std::endl;
    std::cout << "Captured Height Kernel: " << filterHeight << std::endl;
    std::cout << "Captured Width Kernel: " << filterWidth << std::endl;

    Image newImage(3,Matrix(output_height, Array(output_width)) );

    double tempSum = 0;

    for(int m = 0; m < 3; m++){
        for(int i = 0; i < output_height; i++){
            for(int j = 0;j < output_width; j++){


                for(int k = 0; k < filterHeight; k++){
                    for(int z = 0; z < filterWidth; z++){
                        if(samePadding){
                            tempSum += paddedImage[m][i+k][j+z] * filter[k][z];

                        }else{
                            tempSum += image[m][i+k][j+z] * filter[k][z];
                        }

                    }
                }
                newImage[m][i][j] = std::max((double)0., std::min((double)255., tempSum));
                tempSum = 0;
            }
        }
    }

    return newImage;
}