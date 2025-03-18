#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <iostream> 
#include <cuda_runtime.h>
#include <math.h>

using namespace std;

__global__ void matmul_tiled(float* A, float* B, float* C, int batch_size, int N1, int N2, int N3);

#define RANGE 10.0

int main(int argc, const char* argv[]) {
    int N1 = 4;
    int N2 = 2;
    int N3 = 6;
    int batch_size = 8;

    // Allocate host memory
    float *A = (float*)malloc(batch_size * N1 * N2 * sizeof(float));
    float *B = (float*)malloc(batch_size * N2 * N3 * sizeof(float));
    float *C = (float*)malloc(batch_size * N1 * N3 * sizeof(float)); 

    // Initialize A and B with random values
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < N1; ++j) {
            for (int k = 0; k < N2; ++k) {
                A[i * (N1 * N2) + j * N2 + k] = ((float)rand() / (float)(RAND_MAX)) * RANGE;
            }
        }

        for (int k = 0; k < N2; ++k) {
            for (int m = 0; m < N3; ++m) {
                B[i * (N2 * N3) + k * N3 + m] = ((float)rand() / (float)(RAND_MAX)) * RANGE;
            }
        }
    }

    // Allocate device memory
    float *Ad, *Bd, *Cd;
    int Asize = batch_size * N1 * N2 * sizeof(float);
    int Bsize = batch_size * N2 * N3 * sizeof(float);
    int Csize = batch_size * N1 * N3 * sizeof(float);

    cudaMalloc((void**)&Ad, Asize);
    cudaMalloc((void**)&Bd, Bsize);
    cudaMalloc((void**)&Cd, Csize);

    // Copy A and B to device
    cudaMemcpy(Ad, A, Asize, cudaMemcpyHostToDevice);
    cudaMemcpy(Bd, B, Bsize, cudaMemcpyHostToDevice);

    // Launch the kernel
    const int TILE_DIM = 8; 
    
    dim3 threadsPerBlock(TILE_DIM, TILE_DIM, TILE_DIM);

    // Consider updating this as 10 x 10 grid
    dim3 numBlocks((N1 + TILE_DIM-1) / TILE_DIM, (N3 + TILE_DIM-1) / TILE_DIM);   

    matmul_tiled<<<numBlocks, threadsPerBlock>>>(Ad, Bd, Cd, batch_size, N1, N2, N3);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Copy result back to host
    cudaMemcpy(C, Cd, Csize, cudaMemcpyDeviceToHost);

    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < N1; ++j) {
            for (int k = 0; k < N3; ++k) { 
                printf("C[%d][%d][%d] = %f\n", i, j, k, C[k * N1 * N3 + i * N3 + j]);
            }
        }
    }    

    // Free device memory
    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);

    // Free host memory
    free(A);
    free(B);
    free(C);

    return 0;
}

void matmul_cpu(float* A, float* B, float* C, int batch_size, int N1, int N2, int N3) {
    for(int b = 0; b < batch_size; b++) {
        for(int i = 0; i < N1; i++) {
            for(int j = 0; j < N3; j++) {
                float sum = 0.0f;
                for(int k = 0; k < N2; k++) {
                    sum += A[b * N1 * N2 + i * N2 + k] * B[b * N2 * N3 + k * N3 + j];
                }
                C[b * N1 * N3 + i * N3 + j] = sum;
            }
        }
    }
}

__global__ void matmul_tiled(float* A, float* B, float* C, int batch_size, int N1, int N2, int N3) {
    const int TILE_DIM = 8;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    
    int row = blockDim.y * blockIdx.y + ty;
    int col = blockDim.x * blockIdx.x + tx;
    int batch = blockDim.z * blockIdx.z + tz; 
    
    __shared__ float As[TILE_DIM][TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM][TILE_DIM];
    
    float sum = 0.0f;

    int num_tiles = (N2 + TILE_DIM - 1) / TILE_DIM;
    
    for (int tile_index = 0; tile_index < num_tiles; tile_index++) {
        if (batch < batch_size && row < N1 && (tile_index * TILE_DIM + tx) < N2) {
            As[tz][ty][tx] = A[batch * (N1 * N2) + row * N2 + tile_index * TILE_DIM + tx];
        } 
        else {
            As[tz][ty][tx] = 0.0f;
        }
        
        if (batch < batch_size && (tile_index * TILE_DIM + ty) < N2 && col < N3) {
            Bs[tz][ty][tx] = B[batch * (N2 * N3) + (tile_index * TILE_DIM + ty) * N3 + col];
        } 
        else {
            Bs[tz][ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        if (batch < batch_size) {
            for (int k = 0; k < TILE_DIM; k++) {
                sum += As[tz][ty][k] * Bs[tz][k][tx];
            }
        }
        
        __syncthreads();
    }
    
    if (batch < batch_size && row < N1 && col < N3) {
        C[batch * (N1 * N3) + row * N3 + col] = sum;
    }
}