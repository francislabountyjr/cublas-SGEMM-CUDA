﻿/*#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

#include <iostream>
#include <iomanip>
#include <cstdlib>

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

float* getMatrix(const int m, const int ldm);
void printMatrix(const float* matrix, const int m, const int ldm);

int main()
{
    cublasHandle_t handle;

    // prepare input matrices
    float* A, * B, * C;
    int M, N, K;
    float alpha, beta;

    M = 3;
    N = 4;
    K = 7;
    alpha = 1.f;
    beta = 0.f;

    // create cuBLAS handle
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "CUBLAS initialization failed\n";
        return EXIT_FAILURE;
    }

    srand(2021);

    A = getMatrix(K, M);
    B = getMatrix(N, K);
    C = getMatrix(M, N);

    std::cout << "A:\n";
    printMatrix(A, K, M);
    std::cout << "B:\n";
    printMatrix(B, N, K);
    std::cout << "C:\n";
    printMatrix(C, M, N);

    // gemm
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
        M, N, K,
        &alpha,
        A, K,
        B, N,
        &beta,
        C, M);

    cudaDeviceSynchronize();
    std::cout << "C out:\n";
    printMatrix(C, M, N);

    cublasDestroy(handle);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}

float* getMatrix(const int m, const int ldm)
{
    float* pf_matrix = nullptr;
    cudaMallocManaged((void**)&pf_matrix, sizeof(float) * ldm * m);

    for (int j = 0; j < m; j++)
    {
        for (int i = 0; i < ldm; i++)
        {
            pf_matrix[IDX2C(i, j, ldm)] = (float)rand() / RAND_MAX;
        }
    }

    return pf_matrix;
}

void printMatrix(const float* matrix, const int m, const int ldm)
{
    for (int j = 0; j < m; j++)
    {
        for (int i = 0; i < ldm; i++)
        {
            std::cout << std::fixed << std::setw(8) << std::setprecision(4) << matrix[IDX2C(i, j, ldm)];
        }
        std::cout << '\n';
    }
}
*/