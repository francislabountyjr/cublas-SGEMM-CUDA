#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

#include <iostream>
#include <iomanip>
#include <cstdlib>

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

float* getMatrix(const int ldm, const int n);
void printMatrix(const float* matrix, const int ldm, const int n);

int main()
{
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    cudaStream_t stream;

    // prepare input matrices
    float* pf_A, * pf_B, * pf_C;
    float* df_A, * df_B, * df_C;
    int M, N, K;
    float alpha, beta;

    M = 4;
    N = 5;
    K = 6;
    alpha = 1.f;
    beta = 1.f;

    // create cuBLAS handle
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "CUBLAS initialization failed\n";
        return EXIT_FAILURE;
    }

    srand(2021);

    pf_A = getMatrix(K, M);
    pf_B = getMatrix(N, K);
    pf_C = getMatrix(M, N);

    std::cout << "A:\n";
    printMatrix(pf_A, K, M);
    std::cout << "B:\n";
    printMatrix(pf_B, N, K);
    std::cout << "C:\n";
    printMatrix(pf_C, M, N);

    // allocate device memory
    cudaMalloc((void**)&df_A, M * K * sizeof(float));
    cudaMalloc((void**)&df_B, K * N * sizeof(float));
    cudaMalloc((void**)&df_C, M * N * sizeof(float));

    // create stream
    cudaStat = cudaStreamCreate(&stream);

    // asynchronously set cublas matrix
    cublasSetMatrixAsync(M, K, sizeof(*df_A), pf_A, M, df_A, M, stream);
    cublasSetMatrixAsync(K, N, sizeof(*df_B), pf_B, K, df_B, K, stream);
    cublasSetMatrixAsync(M, N, sizeof(*df_C), pf_C, M, df_C, M, stream);

    cublasSetStream(handle, stream);

    // gemm
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K,
        &alpha,
        df_A, M,
        df_B, K,
        &beta,
        df_C, M);

    cublasGetMatrixAsync(M, N, sizeof(*df_C), df_C, M, pf_C, M, stream);

    cudaStreamSynchronize(stream);
    std::cout << "C out:\n";
    printMatrix(pf_C, M, N);

    cublasDestroy(handle);
    cudaStreamDestroy(stream);

    cudaFree(df_A);
    cudaFree(df_B);
    cudaFree(df_C);

    free(pf_A);
    free(pf_B);
    free(pf_C);

    return 0;
}

float* getMatrix(const int ldm, const int n)
{
    float* pf_matrix = (float*)malloc(ldm * n * sizeof(float));

    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < ldm; i++)
        {
            pf_matrix[IDX2C(i, j, ldm)] = (float)rand() / RAND_MAX;
        }
    }

    return pf_matrix;
}

void printMatrix(const float* matrix, const int ldm, const int n)
{
    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < ldm; i++)
        {
            std::cout << std::fixed << std::setw(8) << std::setprecision(4) << matrix[IDX2C(i, j, ldm)];
        }
        std::cout << '\n';
    }
}
