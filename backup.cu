#include <iostream>
#include <vector>
#include <filesystem>
#include <list>
#include <png++/png.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

namespace fs = std::filesystem;

#define BLOCK_SIZE (16u)
#define FILTER_SIZE (5u)
#define TILE_SIZE (12u)

#define CUDA_CHECK_RETURN(value){                               \
    cudaError_t err = value;                                    \
    if( err != cudaSuccess ){                                   \
        fprintf(stderr,"Error %s at line %d in file %s\n",      \
        cudaGetErrorString(err),__LINE__,__FILE__);             \
        exit(1);                                                \
    }}

#include "png.h"

__global__ void processImage(unsigned char*  out, unsigned char* in, size_t pitch, unsigned int width, unsigned int height){
    int x_o = (TILE_SIZE*blockIdx.x) + threadIdx.x;
    int y_o = (TILE_SIZE*blockIdx.y)+threadIdx.y;

    int x_i = x_o - FILTER_SIZE/2;
    int y_i = y_o - FILTER_SIZE/2;
    int sum = 0;

    __shared__ unsigned char sBuffer[BLOCK_SIZE][BLOCK_SIZE];

    if( (x_i >= 0) && (x_i < width) && (y_i >= 0) && (y_i < height)){
        sBuffer[threadIdx.y][threadIdx.x] = in[y_i*pitch+x_i];
    }else{
        sBuffer[threadIdx.y][threadIdx.x] = 0;
    }

    __syncthreads();

    if(threadIdx.x < TILE_SIZE && threadIdx.y < TILE_SIZE){
        for(int r = 0; r < FILTER_SIZE; ++r)
            for(int c = 0; c<FILTER_SIZE; ++c)
                sum += sBuffer[threadIdx.y+r][threadIdx.x+c];


        sum /= FILTER_SIZE* FILTER_SIZE;

        if(x_o < width && y_o < height)
            out[y_o *width + x_o] = sum;
    }


}

int main(){
    std::cout << "Loading Image..." << std::endl;

    png::image<png::rgb_pixel> img("../images/original_image.png");

    unsigned int width = img.get_width();
    unsigned int height = img.get_height();

    int size = width * height * sizeof( unsigned char);

    unsigned char *h_r = (unsigned char*)malloc(size * sizeof(unsigned char));
    unsigned char *h_g = (unsigned char*)malloc(size * sizeof(unsigned char));
    unsigned char *h_b = (unsigned char*)malloc(size * sizeof(unsigned char));

    unsigned char *h_r_n = (unsigned char*)malloc(size * sizeof(unsigned char));
    unsigned char *h_g_n = (unsigned char*)malloc(size * sizeof(unsigned char));
    unsigned char *h_b_n = (unsigned char*)malloc(size * sizeof(unsigned char));

    pngio::pngToRgb3(h_r,h_g,h_b,img);

    unsigned char* d_r_n = NULL;
    unsigned char* d_g_n = NULL;
    unsigned char* d_b_n = NULL;

    CUDA_CHECK_RETURN(cudaMalloc(&d_r_n,size));
    CUDA_CHECK_RETURN(cudaMalloc(&d_g_n,size));
    CUDA_CHECK_RETURN(cudaMalloc(&d_b_n,size));


    unsigned char* d_r = NULL;
    unsigned char* d_g = NULL;
    unsigned char* d_b = NULL;

    size_t pitch_r = 0;
    size_t pitch_g = 0;
    size_t pitch_b = 0;

    CUDA_CHECK_RETURN(cudaMallocPitch(&d_r,&pitch_r,width,height));
    CUDA_CHECK_RETURN(cudaMallocPitch(&d_g,&pitch_g,width,height));
    CUDA_CHECK_RETURN(cudaMallocPitch(&d_b,&pitch_b,width,height));

    CUDA_CHECK_RETURN(cudaMemcpy2D(d_r,pitch_r,h_r,width,width,height,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy2D(d_g,pitch_g,h_g,width,width,height,cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy2D(d_b,pitch_b,h_b,width,width,height,cudaMemcpyHostToDevice));

    dim3 grid_size((width + TILE_SIZE - 1)/TILE_SIZE, (height + TILE_SIZE - 1)/TILE_SIZE);
    dim3 block_size (BLOCK_SIZE,BLOCK_SIZE);

    processImage<<<grid_size,block_size>>>(d_r_n, d_r, pitch_r, width, height);
    processImage<<<grid_size,block_size>>>(d_g_n, d_g, pitch_g, width, height);
    processImage<<<grid_size,block_size>>>(d_b_n, d_b, pitch_b, width, height);

    CUDA_CHECK_RETURN(cudaMemcpy(h_r_n,d_r_n,size,cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(h_g_n,d_g_n,size,cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(h_b_n,d_b_n,size,cudaMemcpyDeviceToHost));

    pngio::rgb3toPng(h_r_n,h_g_n,h_b_n,img);

    img.write("../images/aaa.png");

    CUDA_CHECK_RETURN(cudaFree(d_r));
    CUDA_CHECK_RETURN(cudaFree(d_r_n));

    CUDA_CHECK_RETURN(cudaFree(d_g));
    CUDA_CHECK_RETURN(cudaFree(d_g_n));

    CUDA_CHECK_RETURN(cudaFree(d_b));
    CUDA_CHECK_RETURN(cudaFree(d_b_n));

    free(h_r);
    free(h_r_n);
    free(h_g);
    free(h_g_n);
    free(h_b);
    free(h_b_n);
}

/*__global__ void convolution_3channel(float *input, float *output, int height, int width, int channels, const float* __restrict__ kernel, int kernel_size)
{

    __shared__ float shared_input[BLOCK_SIZE][BLOCK_SIZE][3];

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int z = threadIdx.z;

    // Load input image into shared memory
    if (x < width && y < height)
    {
        shared_input[threadIdx.x][threadIdx.y][z] = input[(y * width + x) * channels + z];
    }
    __syncthreads();

    float sum = 0.0f;
    int half_kernel_size = kernel_size / 2;
    for (int i = -half_kernel_size; i <= half_kernel_size; i++)
    {
        for (int j = -half_kernel_size; j <= half_kernel_size; j++)
        {
            int x_index = threadIdx.x + j;
            int y_index = threadIdx.y + i;

            if (x_index >= 0 && x_index < BLOCK_SIZE && y_index >= 0 && y_index < BLOCK_SIZE)
            {
                sum += shared_input[x_index][y_index][z] * kernel[(i + half_kernel_size) * kernel_size + (j + half_kernel_size)];
            }
            else
            {
                int x_global = x + j;
                int y_global = y + i;

                if (x_global >= 0 && x_global < width && y_global >= 0 && y_global < height)
                {
                    sum += input[(y_global * width + x_global) * channels + z] * kernel[(i + half_kernel_size) * kernel_size + (j + half_kernel_size)];
                }
            }
        }
    }

    if (x < width && y < height)
    {
        output[(y * width + x) * channels + z] = max(0.,min(255.,sum));
    }
}
*/