#include <iostream>
#include <vector>
#include <filesystem>
#include <list>
#include <png++/png.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include "png.h"

namespace fs = std::filesystem;

#define BLOCK_SIZE 16u
#define FILTER_SIZE 3u
#define TILE_SIZE 14u

#define CUDA_CHECK_RETURN(value){                               \
    cudaError_t err = value;                                    \
    if( err != cudaSuccess ){                                   \
        fprintf(stderr,"Error %s at line %d in file %s\n",      \
        cudaGetErrorString(err),__LINE__,__FILE__);             \
        exit(1);                                                \
                                                                \
    }}

char GaussianBlurkernel[FILTER_SIZE * FILTER_SIZE] = {
        1, 2, 1,
        2, 4, 2,
        1, 2, 1
};


char BlurBoxKernel[FILTER_SIZE * FILTER_SIZE] = {
        1, 1, 1,
        1, 1, 1,
        1, 1, 1
};


char SharpenKernel[FILTER_SIZE * FILTER_SIZE] = {
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0
};



__global__ void convolution(unsigned char*  out,  unsigned char* in, size_t pitch,
                            const char*  kernel, const int factor,
                            unsigned int width, unsigned int height){

    int x_o = (TILE_SIZE * blockIdx.x) + threadIdx.x;
    int y_o = (TILE_SIZE * blockIdx.y) + threadIdx.y;

    //center
    int x_i = x_o - (FILTER_SIZE/2);
    int y_i = y_o - (FILTER_SIZE/2);
    float sum = 0;

    __shared__ unsigned char sMem[BLOCK_SIZE][BLOCK_SIZE];

    // inside of image then copy
    if( (x_i >=0) && (x_i < width) && (y_i >=0) && (y_i < height)){
        sMem[threadIdx.y][threadIdx.x] =  in[(y_i*pitch)+x_i];
    }else{
        sMem[threadIdx.y][threadIdx.x] =  0;
    }

    __syncthreads();

    if(threadIdx.x < TILE_SIZE && threadIdx.y < TILE_SIZE){
        for(int r = 0; r < FILTER_SIZE; ++r){
            for(int c = 0; c< FILTER_SIZE; ++c){
                sum += (float)((float)sMem[threadIdx.y+r][threadIdx.x+c] * ((float)kernel[r*FILTER_SIZE+c]/(float)factor));
            }
        }



        if( (x_o < width) && (y_o < height)){
            out[(y_o*width)+x_o] = (int)max(0.,min(255.,sum));
        }
    }




}




//Better: kernel<<<(N+127) / 128, 128>>>( ... )



int main()
{

    cudaFree(0);

    std::vector<std::string> images;
    std::string path = "../images/";
    std::string targetPath = "../filtered_images/";
    for (const auto & entry : fs::directory_iterator(path)){
        images.push_back(entry.path());
        //std::cout << entry.path() << std::endl;
    }



    auto showMatrix = [&](float* matrix,int width, int height){
        for(int i = 0; i < height; i++){
            for(int j = 0 ; j < width; j++){
                std::cout << matrix[i*width+j] << " ";
            }
            std::cout << std::endl;
        }
    };


    //showMatrix(GaussianBlurkernel, KERNEL_SIZE, KERNEL_SIZE);
    //showMatrix(BlurBoxKernel, KERNEL_SIZE,KERNEL_SIZE);



    std::vector<char*> filters {GaussianBlurkernel, BlurBoxKernel, SharpenKernel};
    std::vector<int> factor {16,9,1};
    const std::vector<std::string> filtersName {"GaussianBlur","BoxBlurFilter", "SharpenFilter"};

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    for(int i = 0; i < images.size();i++){
        std::string filePath = images[i];

        std::cout << "Loading image..." << std::endl;

        png::image<png::rgb_pixel> input_image(filePath);
        unsigned int width = input_image.get_width();
        unsigned int height = input_image.get_height();

        std::cout << "Width: " << width << " Height:" << height << std::endl;

        unsigned int size = width * height * sizeof(unsigned char);

        //Red, green, blue pixels from input image
        unsigned char* input_image_pixel_red = (unsigned char*)malloc(size);
        unsigned char* input_image_pixel_green = (unsigned char*)malloc(size);
        unsigned char* input_image_pixel_blue = (unsigned char*)malloc(size);

        //Output from kernel function (convolution)
        unsigned char *output_image_pixel_red = (unsigned char *) malloc(size);
        unsigned char *output_image_pixel_green = (unsigned char *) malloc(size);
        unsigned char *output_image_pixel_blue = (unsigned char *) malloc(size);

        //Store pixel in variables
        pngio::pngToRgb3(input_image_pixel_red, input_image_pixel_green,input_image_pixel_blue,input_image);


        // Allocate memory on GPU
        unsigned char* input_data_gpu_red;
        unsigned char* input_data_gpu_green;
        unsigned char* input_data_gpu_blue;

        size_t pitch_r = 0;
        size_t pitch_g = 0;
        size_t pitch_b = 0;


        cudaMallocPitch(&input_data_gpu_red,&pitch_r,width,height);
        cudaMallocPitch(&input_data_gpu_green,&pitch_g,width,height);
        cudaMallocPitch(&input_data_gpu_blue,&pitch_b,width,height);

        cudaMemcpy2D(input_data_gpu_red,pitch_r,input_image_pixel_red,width,width,height,cudaMemcpyHostToDevice);
        cudaMemcpy2D(input_data_gpu_green,pitch_g,input_image_pixel_green,width,width,height,cudaMemcpyHostToDevice);
        cudaMemcpy2D(input_data_gpu_blue,pitch_b,input_image_pixel_blue,width,width,height,cudaMemcpyHostToDevice);


        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize((width + TILE_SIZE -1)/TILE_SIZE,(height + TILE_SIZE - 1)/TILE_SIZE);

        unsigned char* output_data_gpu_red;
        unsigned char* output_data_gpu_green;
        unsigned char* output_data_gpu_blue;

        cudaMalloc(&output_data_gpu_red,size);
        cudaMalloc(&output_data_gpu_green,size);
        cudaMalloc(&output_data_gpu_blue,size);

        for(int j = 0; j < 1;j++) {

            char* kernel = NULL;
            cudaMalloc(&kernel,FILTER_SIZE*FILTER_SIZE);
            cudaMemcpy(kernel,filters[j],FILTER_SIZE*FILTER_SIZE,cudaMemcpyHostToDevice);

            const int factorKernel = factor[j];

            convolution<<<gridSize, blockSize>>>(output_data_gpu_red, input_data_gpu_red, pitch_r, kernel, factorKernel, width, height);
            convolution<<<gridSize, blockSize>>>(output_data_gpu_green, input_data_gpu_green, pitch_g,kernel, factorKernel, width, height);
            convolution<<<gridSize, blockSize>>>(output_data_gpu_blue, input_data_gpu_blue, pitch_b,kernel, factorKernel, width, height);

            cudaDeviceSynchronize();

            cudaMemcpy(output_image_pixel_red, output_data_gpu_red, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(output_image_pixel_green, output_data_gpu_green, size, cudaMemcpyDeviceToHost);
            cudaMemcpy(output_image_pixel_blue, output_data_gpu_blue, size, cudaMemcpyDeviceToHost);

            // Convert output data to PNG image
            pngio::rgb3toPng(output_image_pixel_red,output_image_pixel_green,output_image_pixel_blue,input_image);

            std::string targetFilePathName = targetPath + filtersName[j] + "_" +  std::string(fs::path(filePath).filename());
            input_image.write(targetFilePathName);
            std::cout << "Saved in " << targetFilePathName << std::endl;

            cudaFree(kernel);
        }






        cudaFree(output_data_gpu_red);
        cudaFree(output_data_gpu_green);
        cudaFree(output_data_gpu_blue);
        cudaFree(input_data_gpu_red);
        cudaFree(input_data_gpu_green);
        cudaFree(input_data_gpu_blue);

        free(output_image_pixel_red);
        free(output_image_pixel_green);
        free(output_image_pixel_blue);
        free(input_image_pixel_red);
        free(input_image_pixel_blue);
        free(input_image_pixel_green);


        /*
        for(int j = 0; j < filters.size();j ++){

            cudaMemcpy(kernel_gpu, filters[j], KERNEL_SIZE * KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            convolution_3channel<<<grid_size, block_size>>>(input_data_gpu, output_data_gpu, width, height, 3, kernel_gpu, KERNEL_SIZE);
            cudaMemcpy(input_data, output_data_gpu, width * height * 3 * sizeof(float), cudaMemcpyDeviceToHost);

            // Convert output data to PNG image
            idx = 0;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int red = (int)(input_data[idx++] );
                    int green = (int)(input_data[idx++] );
                    int blue = (int)(input_data[idx++] );
                    output_image[y][x] = png::rgb_pixel(red, green, blue);
                }
            }

            std::string targetFilePathName = targetPath + filtersName[j] + "_" + std::string(fs::path( filePath ).filename());
            output_image.write(targetFilePathName);
            std::cout << "Saved in " << targetFilePathName << std::endl;
        }










        cudaFree(input_data_gpu);
        cudaFree(output_data_gpu);
        cudaFree(kernel_gpu);
        /*
        for(int j = 0; j < filters.size();j ++){
            std::cout << "Convoluting with "+filtersName[j]+" kernel matrix... " << std::endl;
            Image filteredImage = kip.convolute(image,filters[j],true);
            std::cout << "Convolution completed!" << std::endl;

            std::string targetFilePathName = targetPath + filtersName[j] + "_" + std::string(fs::path( filePath ).filename());
            kip.saveImage(filteredImage, targetFilePathName.c_str());
            filteredImage.clear();

        }

         */


    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    int timeDifference = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

    std::cout << "Elapsed Time: " << timeDifference << std::endl;


    return 0;
}