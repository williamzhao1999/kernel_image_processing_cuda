//
// Created by pc23wilzha on 26/03/23.
//

#include "png.h"

void pngio::pngToRgb3(unsigned char* r, unsigned char* g, unsigned char* b,png::image<png::rgb_pixel>& img){
    int idx = 0;
    const int height = img.get_height();
    const int width = img.get_width();

    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            r[(i*width)+j] = img[i][j].red;
            g[(i*width)+j] = img[i][j].green;
            b[(i*width)+j] = img[i][j].blue;
            idx++;
        }
    }
}
void pngio::rgb3toPng(unsigned char* r, unsigned char* g, unsigned char* b,png::image<png::rgb_pixel>& img){
    int idx = 0;
    const int height = img.get_height();
    const int width = img.get_width();


    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            img[i][j].red = r[idx];
            img[i][j].green = g[idx];
            img[i][j].blue = b[idx];
            idx++;
        }
    }
}