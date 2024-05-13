#ifndef IM2COL_H
#define IM2COL_H
#define BLACK_NUM -9999.99

void im2col_cpu(float* data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_col);

int black_im2col_cpu(float* data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_col, int* black_pixel, int black_pixel_size);

#ifdef GPU

void im2col_gpu(float *im,
         int channels, int height, int width,
         int ksize, int stride, int pad,float *data_col);

#endif
#endif
