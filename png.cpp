//
// Created by pc23wilzha on 26/03/23.
//

#include "png.h"

void pngio::pngToRgb3(unsigned char* r, unsigned char* g, unsigned char* b,png::image<png::rgb_pixel>& img){
    for(int i = 0; i < img.get_height(); i++){
        for(int j = 0; j < img.get_width(); j++){
            r[i*img.get_height()+j] = img[i][j].red;
            g[i*img.get_height()+j] = img[i][j].green;
            b[i*img.get_height()+j] = img[i][j].blue;
        }
    }
}
void pngio::rgb3toPng(unsigned char* r, unsigned char* g, unsigned char* b,png::image<png::rgb_pixel>& img){
    for(int i = 0; i < img.get_height(); i++){
        for(int j = 0; j < img.get_width(); j++){
            img[i][j].red = r[i*img.get_height()+j];
            img[i][j].green = g[i*img.get_height()+j] ;
            img[i][j].blue = b[i*img.get_height()+j];
        }
    }
}