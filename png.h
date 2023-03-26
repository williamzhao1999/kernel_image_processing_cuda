//
// Created by pc23wilzha on 26/03/23.
//

#ifndef KERNEL_IMAGE_PROCESSING_PNG_H
#define KERNEL_IMAGE_PROCESSING_PNG_H

#include <png++/png.hpp>

class pngio {

public:
    static void pngToRgb3(unsigned char* r, unsigned char* g, unsigned char* b,png::image<png::rgb_pixel>& img);
    static void rgb3toPng(unsigned char* r, unsigned char* g, unsigned char* b,png::image<png::rgb_pixel>& img);
};


#endif //KERNEL_IMAGE_PROCESSING_PNG_H
