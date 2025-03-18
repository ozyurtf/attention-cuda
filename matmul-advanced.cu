#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

// 2D Thread Block Implementation
__global__ void matmul_tiled_2d(float* A, float* B, float* C, int batch_size, int N1, int N2, int N3) {
    const int TILE_DIM = 16;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;
    int batch = blockIdx.z;
    
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    
    float sum = 0.0f;
    
    for (int t = 0; t < (N2 + TILE_DIM - 1) / TILE_DIM; t++) {
        if (row < N1 && (t * TILE_DIM + tx) < N2) {
            As[ty][tx] = A[batch * (N1 * N2) + row * N2 + t * TILE_DIM + tx];
        } 
        else {
            As[ty][tx] = 0.0f;
        }
        
        if ((t * TILE_DIM + ty) < N2 && col < N3) {
            Bs[ty][tx] = B[batch * (N2 * N3) + (t * TILE_DIM + ty) * N3 + col];
        } 
        else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        for (int k = 0; k < TILE_DIM; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < N1 && col < N3) {
        C[batch * (N1 * N3) + row * N3 + col] = sum;
    }
}

__global__ void matmul_tiled_3d(float* A, float* B, float* C, int batch_size, int N1, int N2, int N3) {
    const int TILE_DIM = 8;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    
    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;
    int batch = blockIdx.z * blockDim.z + tz;
    
    __shared__ float As[8][8][8];
    __shared__ float Bs[8][8][8];
    
    float sum = 0.0f;
    
    for (int t = 0; t < (N2 + TILE_DIM - 1) / TILE_DIM; t++) {
        if (batch < batch_size && row < N1 && (t * TILE_DIM + tx) < N2) {
            As[tz][ty][tx] = A[batch * (N1 * N2) + row * N2 + t * TILE_DIM + tx];
        } 
        else {
            As[tz][ty][tx] = 0.0f;
        }
        
        if (batch < batch_size && (t * TILE_DIM + ty) < N2 && col < N3) {
            Bs[tz][ty][tx] = B[batch * (N2 * N3) + (t * TILE_DIM + ty) * N3 + col];
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

__global__ void matmul_nested_tiled_3d(float* A, float* B, float* C, int batch_size, int N1, int N2, int N3) {
    const int tile_width_y = 8;  // For dimension N1
    const int tile_width_x = 8;  // For dimension N2
    const int tile_width_z = 8;  // For dimension N3
    
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;
    
    int bid_x = blockIdx.x;
    int bid_y = blockIdx.y;
    int bid_z = blockIdx.z;
    
    // Shared memory for tiles
    __shared__ float sh_A[8][8][8];
    __shared__ float sh_B[8][8][8];
    
    float value = 0.0f;
    
    // Triple nested loop for tiling in all dimensions
    for (int tile_ind_y = 0; tile_ind_y < ceil((float)N1/tile_width_y); tile_ind_y++) {
        for (int tile_ind_x = 0; tile_ind_x < ceil((float)N2/tile_width_x); tile_ind_x++) {
            for (int tile_ind_z = 0; tile_ind_z < ceil((float)N3/tile_width_z); tile_ind_z++) {
                
                // Calculate global indices for this tile
                int y_index = tile_ind_y * tile_width_y + tid_y;
                int x_index = tile_ind_x * tile_width_x + tid_x;
                int z_index = tile_ind_z * tile_width_z + tid_z;
                
                // Load data into shared memory
                if (bid_z < batch_size && 
                    y_index < N1 && 
                    x_index < N2) {
                    sh_A[tid_z][tid_y][tid_x] = A[bid_z * N1 * N2 + y_index * N2 + x_index];
                } 
                else {
                    sh_A[tid_z][tid_y][tid_x] = 0.0f;
                }
                
                if (bid_z < batch_size && 
                    x_index < N2 && 
                    z_index < N3) {
                    sh_B[tid_z][tid_y][tid_x] = B[bid_z * N2 * N3 + x_index * N3 + z_index];
                } 
                else {
                    sh_B[tid_z][tid_y][tid_x] = 0.0f;
                }
                
                __syncthreads();
                
                // Compute partial dot products for this tile
                for (int k = 0; k < tile_width_x; k++) {
                    if (y_index < N1 && z_index < N3 && k < N2) {
                        value += sh_A[tid_z][tid_y][k] * 
                                sh_B[k][tid_y][tid_x];
                    }
                }
                
                __syncthreads();
            }
        }
    }
    
    // Write final result
    int y_out = bid_y * blockDim.y + tid_y;
    int z_out = bid_x * blockDim.x + tid_x;
    
    if (bid_z < batch_size && 
        y_out < N1 && 
        z_out < N3) {
        C[bid_z * N1 * N3 + y_out * N3 + z_out] = value;
    }
}

// Function to print matrix
void printMatrix(float* matrix, int batch, int rows, int cols, const char* name) {
    printf("\n%s for batch %d:\n", name, batch);
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            printf("%8.2f ", matrix[batch * rows * cols + i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// CPU verification
void matmulCPU(float* A, float* B, float* C, int batch_size, int N1, int N2, int N3) {
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

int main() {
    const int N1 = 50;
    const int N2 = 200;
    const int N3 = 150;
    const int batch_size = 100;
    
    size_t size_A = batch_size * N1 * N2 * sizeof(float);
    size_t size_B = batch_size * N2 * N3 * sizeof(float);
    size_t size_C = batch_size * N1 * N3 * sizeof(float);
    
    // Allocate host memory
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C_2d = (float*)malloc(size_C);
    float *h_C_3d = (float*)malloc(size_C);
    float *h_C_cpu = (float*)malloc(size_C);
    
    // Initialize matrices with random values
    srand(42);  // For reproducibility
    for(int b = 0; b < batch_size; b++) {
        for(int i = 0; i < N1; i++) {
            for(int j = 0; j < N2; j++) {
                h_A[b * N1 * N2 + i * N2 + j] = (float)(rand()) / RAND_MAX;
            }
        }
        for(int i = 0; i < N2; i++) {
            for(int j = 0; j < N3; j++) {
                h_B[b * N2 * N3 + i * N3 + j] = (float)(rand()) / RAND_MAX;
            }
        }
    }
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    // Copy input matrices to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    // Launch 2D thread block kernel
    dim3 threads_2d(16, 16);
    dim3 blocks_2d((N3 + 15) / 16, (N1 + 15) / 16, batch_size);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    printf("\nRunning 2D thread block version...\n");
    cudaEventRecord(start);
    matmul_tiled_2d<<<blocks_2d, threads_2d>>>(d_A, d_B, d_C, batch_size, N1, N2, N3);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds_2d = 0;
    cudaEventElapsedTime(&milliseconds_2d, start, stop);
    
    // Copy result back to host
    cudaMemcpy(h_C_2d, d_C, size_C, cudaMemcpyDeviceToHost);
    
    // Launch 3D thread block kernel
    dim3 threads_3d(8, 8, 8);
    dim3 blocks_3d((N3 + 7) / 8, (N1 + 7) / 8, (batch_size + 7) / 8);
    
    printf("Running 3D thread block version...\n");
    cudaEventRecord(start);
    matmul_tiled_3d<<<blocks_3d, threads_3d>>>(d_A, d_B, d_C, batch_size, N1, N2, N3);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds_3d = 0;
    cudaEventElapsedTime(&milliseconds_3d, start, stop);
    
    // Copy result back to host
    cudaMemcpy(h_C_3d, d_C, size_C, cudaMemcpyDeviceToHost);
    
    // CPU verification
    printf("Running CPU verification...\n");
    cudaEventRecord(start);
    matmulCPU(h_A, h_B, h_C_cpu, batch_size, N1, N2, N3);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); 

    float milliseconds_cpu = 0;
    cudaEventElapsedTime(&milliseconds_cpu, start, stop);    
    
    // Print timing results
    printf("\nTiming Results:\n");
    printf("CPU Version: %.3f ms\n", milliseconds_cpu);
    printf("2D Thread Block Version: %.3f ms\n", milliseconds_2d);
    printf("3D Thread Block Version: %.3f ms\n", milliseconds_3d);
    
    // Print sample results and verify
    printf("\nPrinting sample results for batch 0:\n");
    printf("First 5x5 elements of matrices:\n");
    
    // Print input matrices
    for(int i = 0; i < 5; i++) {
        for(int j = 0; j < 5; j++) {
            printf("%8.2f ", h_A[i * N2 + j]);
        }
        printf("\n");
    }
    printf("\nMatrix B:\n");
    for(int i = 0; i < 5; i++) {
        for(int j = 0; j < 5; j++) {
            printf("%8.2f ", h_B[i * N3 + j]);
        }
        printf("\n");
    }
    
    // Print and compare results
    printf("\n2D Thread Block Result:\n");
    for(int i = 0; i < 5; i++) {
        for(int j = 0; j < 5; j++) {
            printf("%8.2f ", h_C_2d[i * N3 + j]);
        }
        printf("\n");
    }
    
    printf("\n3D Thread Block Result:\n");
    for(int i = 0; i < 5; i++) {
        for(int j = 0; j < 5; j++) {
            printf("%8.2f ", h_C_3d[i * N3 + j]);
        }
        printf("\n");
    }
    
    printf("\nCPU Result:\n");
    for(int i = 0; i < 5; i++) {
        for(int j = 0; j < 5; j++) {
            printf("%8.2f ", h_C_cpu[i * N3 + j]);
        }
        printf("\n");
    }
    
    // Verify results
    float max_error_2d = 0.0f;
    float max_error_3d = 0.0f;
    for(int i = 0; i < batch_size * N1 * N3; i++) {
        max_error_2d = max(max_error_2d, abs(h_C_2d[i] - h_C_cpu[i]));
        max_error_3d = max(max_error_3d, abs(h_C_3d[i] - h_C_cpu[i]));
    }
    printf("\nMaximum Error:\n");
    printf("2D Thread Block Version: %e\n", max_error_2d);
    printf("3D Thread Block Version: %e\n", max_error_3d);
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_2d);
    free(h_C_3d);
    free(h_C_cpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}