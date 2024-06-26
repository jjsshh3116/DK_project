#ifndef GEMM_H
#define GEMM_H

#include "darknet.h"


void gemm_bin(int M, int N, int K, float ALPHA,
              char  *A, int lda,
              float *B, int ldb,
              float *C, int ldc);

void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
          float *A, int lda,
          float *B, int ldb,
          float BETA,
          float *C, int ldc);
void gemm_diff(int TA, int TB, int M, int N, int K, float ALPHA,
               float *A, int lda,
               float *B, int ldb,
               float BETA,
               float *C, int ldc);

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
              float *A, int lda,
              float *B, int ldb,
              float BETA,
              float *C, int ldc);
void gemm_cpu_diff(int TA, int TB, int M, int N, int K, float ALPHA,
                   float *A, int lda,
                   float *B, int ldb,
                   float BETA,
                   float *C, int ldc);

void black_gemm_nn(int M, int N, int K, float ALPHA,
             float *A, int lda,
             float *B, int ldb,
             float *C, int ldc,
             black_pixels *black_in_TEE,
             float *pixel_data);

#ifdef GPU
void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA,
              float *A_gpu, int lda,
              float *B_gpu, int ldb,
              float BETA,
              float *C_gpu, int ldc);

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA,
              float *A, int lda,
              float *B, int ldb,
              float BETA,
              float *C, int ldc);
#endif
#endif
