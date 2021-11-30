// LUStage 1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <cuda_runtime.h>

// #include <chrono>
// #include <sys/time.h>
#include <time.h>       /* time */

#include <stdlib.h>     /* srand, rand */
#include <iostream>
#include <algorithm>
#include <string>
#include <math.h>



// #define PRINT_MATRIX
cudaStream_t* gpuStreams;
uint32_t N_THREADS_PER_BLOCK; // always less than 1024
uint32_t SHARED_MEMORY_SIZE; // SHARED_MEMORY_SIZE = N_THREADS_PER_BLOCK * N, N >= 1 and integer
uint32_t N_STREAM;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    }
}



void printMatrix(const float matrix[], const uint32_t iSize, const uint32_t jSize)
{
    uint32_t index = 0;
    for (uint32_t i = 0; i < iSize; i++)
    {
        for (uint32_t j = 0; j < jSize; j++)
        {
            printf("  %f", matrix[index]);
            index++;
        }
        printf("\n");
    }
}



float get5HundredthsRandom(void)
{
    return  (0.100 * (float)rand() / (float)RAND_MAX) - 0.05;
}



void print_device_properties(void)
{
    int i, deviceCount;
    cudaDeviceProp deviceProp;
    cudaGetDeviceCount(&deviceCount);
    printf("------------------------------------------------------------\n");
    printf("Number of GPU devices found = %d\n", deviceCount);
    for (i = 0; i < deviceCount; ++i) {
        cudaGetDeviceProperties(&deviceProp, i);
        printf("[Device: %1d] Compute Capability %d.%d.\n", i, deviceProp.major, deviceProp.minor);
        printf(" ... multiprocessor count  = %d\n", deviceProp.multiProcessorCount);
        printf(" ... max threads per multiprocessor = %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf(" ... max threads per block = %d\n", deviceProp.maxThreadsPerBlock);
        printf(" ... max block dimension   = %d, %d, %d (along x, y, z)\n",
            deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf(" ... max grid size         = %d, %d, %d (along x, y, z)\n",
            deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf(" ... warp size             = %d\n", deviceProp.warpSize);
        printf(" ... clock rate            = %d MHz\n", deviceProp.clockRate / 1000);
    }
    printf("------------------------------------------------------------\n");
}




uint32_t incrementStreamIndex(uint32_t streamIndex)
{
    streamIndex++;
    if (streamIndex >= N_STREAM)
    {
        streamIndex = 0;
    }
    return streamIndex;
}


////////// Initial Computations //////////
__global__ void generateCaptialKOnGpuStep1(float* matrixKCaptiaGpu, float* gridPositionGpu, uint32_t gridM, uint32_t i, uint32_t j, uint32_t k)
{
    float xDistancePart = gridPositionGpu[i] - gridPositionGpu[k];
    xDistancePart = xDistancePart * xDistancePart;
    float  yDistancePart = gridPositionGpu[j] - gridPositionGpu[threadIdx.x];
    yDistancePart = yDistancePart * yDistancePart;
    float totalDistance = yDistancePart + xDistancePart;
    float kValue = expf(-totalDistance);
    matrixKCaptiaGpu[i * gridM * gridM * gridM + j * gridM * gridM + k * gridM + threadIdx.x] = kValue;
    //printf("tId: %d, i: %d, j: %d, k:%d, g[i]: %f, g[j]: %f, g[k]: %f, g[id]: %f, x: %f, y: %f, td: %f, k: %f, in: %d, s: %f\n", 
    //    threadIdx.x, i, j, k, gridPositionGpu[i], gridPositionGpu[j], gridPositionGpu[k], gridPositionGpu[threadIdx.x],
    //    xDistancePart, yDistancePart, totalDistance, kValue, i * gridM * gridM * gridM + j * gridM * gridM + k * gridM + threadIdx.x,
    //    matrixKCaptiaGpu[i * gridM * gridM * gridM + j * gridM * gridM + k * gridM + threadIdx.x]);
}




void generateCaptialKOnGpu(float* matrixKCaptiaGpu, float* gridPositionGpu, uint32_t gridM)
{
    uint32_t streamIndex = 0;
    for (uint32_t i = 0; i < gridM; i++)
    {
        for (uint32_t j = 0; j < gridM; j++)
        {
            for (uint32_t k = 0; k < gridM; k++)
            {
                // printf("nThreadsNeeded: %d, i: %d, j: %d, k:%d.\n", gridM, i, j, k);
                generateCaptialKOnGpuStep1 << <1, gridM, 0, gpuStreams[streamIndex] >> > (matrixKCaptiaGpu, gridPositionGpu, gridM, i, j, k);
                streamIndex = incrementStreamIndex(streamIndex);
                gpuErrchk(cudaPeekAtLastError());
            }
        }
    }
}



__global__ void addNoiseOnGpuStep1(float* matrixKCaptiaGpu, uint32_t n, uint32_t iStart)
{
    uint32_t i = iStart + threadIdx.x;
    matrixKCaptiaGpu[i * n + i] += 0.01;
}



void addNoiseOnGpu(float* matrixKCaptiaGpu, uint32_t gridM, uint32_t maxNThreadsPerBlock)
{
    uint32_t streamIndex = 0;
    uint32_t nThreadsNeeded = gridM * gridM;
    uint32_t n = gridM * gridM;
    for (uint32_t i = 0; i < gridM * gridM;)
    {
        if (nThreadsNeeded > maxNThreadsPerBlock)
        {
            addNoiseOnGpuStep1 << <1, maxNThreadsPerBlock, 0, gpuStreams[streamIndex] >> > (matrixKCaptiaGpu, n, i);
            gpuErrchk(cudaPeekAtLastError());
            streamIndex = incrementStreamIndex(streamIndex);
            nThreadsNeeded -= maxNThreadsPerBlock;
            i += maxNThreadsPerBlock;
        }
        else
        {
            addNoiseOnGpuStep1 << <1, nThreadsNeeded, 0, gpuStreams[streamIndex] >> > (matrixKCaptiaGpu, n, i);
            gpuErrchk(cudaPeekAtLastError());
            streamIndex = incrementStreamIndex(streamIndex);
            nThreadsNeeded -= maxNThreadsPerBlock;
            i = gridM * gridM;
        }
    }
}




__global__ void generateKTransposeOnGpuStep1(float* matrixKTransposeGpu, float* gridPositionGpu, uint32_t gridM, float predictionPointX, float predictionPointY, uint32_t i)
{
    float xDistancePart = gridPositionGpu[i] - predictionPointX;
    xDistancePart = xDistancePart * xDistancePart;
    float yDistancePart = gridPositionGpu[threadIdx.x] - predictionPointY;
    yDistancePart = yDistancePart * yDistancePart;
    float totalDistance = yDistancePart + xDistancePart;
    float kValue = exp(-totalDistance);
    matrixKTransposeGpu[i * gridM + threadIdx.x] = kValue;
}



void generateKTransposeOnGpu(float* matrixKTransposeGpu, float* gridPositionGpu, uint32_t gridM, float predictionPointX, float predictionPointY)
{
    uint32_t streamIndex = 0;
    for (uint32_t i = 0; i < gridM; i++)
    {
        generateKTransposeOnGpuStep1 << <1, gridM, 0, gpuStreams[streamIndex] >> > (matrixKTransposeGpu, gridPositionGpu, gridM, predictionPointX, predictionPointY, i);
        streamIndex = incrementStreamIndex(streamIndex);
        gpuErrchk(cudaPeekAtLastError());
    }
}
////////// END: Initial Computations //////////








////////// LU Computation //////////
// iStart is included but iEnd is not included in the range
__global__ void GpuLUCumputeRow(float* matrix, const uint32_t n, uint32_t k, uint32_t iStart, uint32_t jStart, uint32_t nColumn)
{
    // https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    // https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    // https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/#:~:text=x%20and%203.,shared%20memory%20%2F%2048KB%20L1%20cache.

    uint32_t i = iStart + threadIdx.x;
    float toBeL = matrix[i * n + k] / matrix[k * n + k];

    for (uint32_t j = jStart, columnIndex = 0; columnIndex < nColumn; j++, columnIndex++)
    {
        matrix[i * n + j] -= (toBeL * matrix[k * n + j]);
    }
}



__global__ void GpuLUCumputeColumn(float* matrix, const uint32_t n, uint32_t k, uint32_t iStart)
{
    uint32_t i = iStart + threadIdx.x;
    float toBeL = matrix[i * n + k] / matrix[k * n + k];
    matrix[i * n + k] = toBeL;
}



void calculateNColumnValues(uint32_t& nColumn, const uint32_t n, const uint32_t j)
{
    nColumn = n - j;
    if (nColumn > SHARED_MEMORY_SIZE)
    {
        nColumn = SHARED_MEMORY_SIZE;
    }
}



void computeLUMatrixGpuCompact(float* matrix, const uint32_t n, uint32_t maxNThreadsPerBlock)
{
    for (uint32_t k = 0; k < n - 1; k++)
    {
        uint32_t nThreadsNeeded = n - (k + 1);
        uint32_t streamIndex = 0;

        for (uint32_t i = k + 1; i < n;)
        {
            if (nThreadsNeeded >= maxNThreadsPerBlock) // 1024 because it is maximum thread allowed per Multiprocessor
            {
                for (uint32_t j = k + 1; j < n; j += SHARED_MEMORY_SIZE)
                {
                    uint32_t nColumn;
                    calculateNColumnValues(nColumn, n, j);
                    // printf("if: nThreadsNeeded: %d, k: %d, i: %d, j:%d, nColumn: %d, nColumnForMemoryRead: %d\n", nThreadsNeeded, k, i, j, nColumn, nColumnForMemoryRead);
                    GpuLUCumputeRow << <1, maxNThreadsPerBlock, 0, gpuStreams[streamIndex] >> > (matrix, n, k, i, j, nColumn);
                    streamIndex = incrementStreamIndex(streamIndex);
                    gpuErrchk(cudaPeekAtLastError());
                }
                i += maxNThreadsPerBlock;
                nThreadsNeeded -= maxNThreadsPerBlock;
            }
            else
            {
                for (uint32_t j = k + 1; j < n;)
                {
                    uint32_t nColumn, nColumnForMemoryRead;
                    calculateNColumnValues(nColumn, n, j);
                    // printf("else: nThreadsNeeded: %d, k: %d, i: %d, j:%d, nColumn: %d, nColumnForMemoryRead: %d\n", nThreadsNeeded, k, i, j, nColumn,nColumnForMemoryRead);
                    GpuLUCumputeRow << <1, nThreadsNeeded, 0, gpuStreams[streamIndex] >> > (matrix, n, k, i, j, nColumn);
                    streamIndex = incrementStreamIndex(streamIndex);
                    gpuErrchk(cudaPeekAtLastError());

                    j += nColumn;
                }
                i = n;
                nThreadsNeeded = 0;
            }
        }
        cudaDeviceSynchronize();
        gpuErrchk(cudaPeekAtLastError());
    }



    uint32_t streamIndex = 0;
    for (uint32_t k = 0; k < n - 1; k++)
    {
        uint32_t nThreadsNeeded = n - (k + 1);
        for (uint32_t iStart = k + 1; iStart < n;)
        {
            if (nThreadsNeeded >= maxNThreadsPerBlock) // 1024 because it is maximum thread allowed per Multiprocessor
            {
                GpuLUCumputeColumn << <1, maxNThreadsPerBlock, 0, gpuStreams[streamIndex] >> > (matrix, n, k, iStart);
                streamIndex = incrementStreamIndex(streamIndex);
                gpuErrchk(cudaPeekAtLastError());
                nThreadsNeeded -= maxNThreadsPerBlock;
                iStart += maxNThreadsPerBlock;
            }
            else
            {
                GpuLUCumputeColumn << <1, nThreadsNeeded, 0, gpuStreams[streamIndex] >> > (matrix, n, k, iStart);
                streamIndex = incrementStreamIndex(streamIndex);
                gpuErrchk(cudaPeekAtLastError());
                nThreadsNeeded = 0;
                iStart = n;
            }
        }
    }
    cudaDeviceSynchronize();
}

////////// END: LU Computation //////////









////////// L Substitution //////////
__global__ void substitutionForLGPUStep1(float compactMatrix_[], float matrixY_[], uint32_t n, uint32_t i, uint32_t jStart)
{
    uint32_t myIndexInCompactMatrix = i * n + jStart + threadIdx.x;

    compactMatrix_[myIndexInCompactMatrix] = (compactMatrix_[myIndexInCompactMatrix] * matrixY_[threadIdx.x + jStart]);
    __syncthreads();

    for (uint32_t nReductionElements = blockDim.x; nReductionElements > 1; nReductionElements /= 2)
    {
        if (nReductionElements % 2 != 0)
        {
            if (threadIdx.x == nReductionElements - 3) // we ask second last thread to do this so that we do not have to sync
            {
                compactMatrix_[myIndexInCompactMatrix] += compactMatrix_[myIndexInCompactMatrix + 2];
            }
            nReductionElements--;
        }

        if (threadIdx.x < nReductionElements && threadIdx.x % 2 == 0)
        {
            float result = compactMatrix_[myIndexInCompactMatrix] + compactMatrix_[myIndexInCompactMatrix + 1];
            __syncthreads();
            compactMatrix_[i * n + jStart + (threadIdx.x / 2)] = result;
            __syncthreads();
        }
    }
}



__global__ void substitutionForLGPUStep2(float compactMatrix_[], uint32_t n, uint32_t i, uint32_t nMergresNeeded, uint32_t distanceBetweenElements)
{
    uint32_t myIndexInCompactMatrix = i * n + (distanceBetweenElements * 2 * threadIdx.x);

    if (nMergresNeeded % 2 != 0)
    {
        if (threadIdx.x == blockDim.x - 1) // we ask second last thread to do this so that we do not have to sync
        {
            compactMatrix_[myIndexInCompactMatrix] += compactMatrix_[myIndexInCompactMatrix + (distanceBetweenElements * 2)];
        }
        nMergresNeeded--;
    }

    float result = compactMatrix_[myIndexInCompactMatrix] + compactMatrix_[myIndexInCompactMatrix + distanceBetweenElements];
    __syncthreads();
    compactMatrix_[i * n + threadIdx.x] = result;
    __syncthreads();
    myIndexInCompactMatrix = i * n + threadIdx.x;


    for (uint32_t nReductionElements = blockDim.x; nReductionElements > 1; nReductionElements /= 2)
    {
        if (nReductionElements % 2 != 0)
        {
            if (threadIdx.x == nReductionElements - 3) // we ask second last thread to do this so that we do not have to sync
            {
                compactMatrix_[myIndexInCompactMatrix] += compactMatrix_[myIndexInCompactMatrix + 2];
            }
            nReductionElements--;
        }

        if (threadIdx.x < nReductionElements && threadIdx.x % 2 == 0)
        {
            float result = compactMatrix_[myIndexInCompactMatrix] + compactMatrix_[myIndexInCompactMatrix + 1];
            __syncthreads();
            compactMatrix_[i * n + (threadIdx.x / 2)] = result;
            __syncthreads();
        }
    }
}


__global__ void substitutionForLGPUStep3(float matrixB_[], float matrixY_[], uint32_t i)
{
    matrixY_[i] = matrixB_[i];
}



__global__ void substitutionForLGPUStep4(float compactMatrix_[], float matrixB_[], float matrixY_[], uint32_t n, uint32_t i)
{
    uint32_t myIndexInCompactMatrix = i * n;
    matrixY_[i] = (matrixB_[i] - compactMatrix_[myIndexInCompactMatrix]);
}




// AX = B
// A = LU
// LUX = B
// We can solve that by solving the 2 below
// 1) LY = B
// 2) UX = Y
// In this function solving for 1)
// size of compactMatrix is nXn while B and Y are nX1,  Y is unkown
void substitutionForL(float compactMatrix_[], float matrixB_[], float matrixY_[], const uint32_t n, uint32_t maxNThreadsPerBlock)
{
    for (uint32_t i = 0; i < n; i++)
    {
        uint32_t nThreadsNeeded = i;
        uint32_t streamIndex = 0;
        for (uint32_t j = 0; j < i;)
        {
            if (nThreadsNeeded >= maxNThreadsPerBlock) // 1024 because it is maximum thread allowed per Multiprocessor
            {
                // printf("if: nThreadsNeeded: %d, i: %d, j:%d.\n", nThreadsNeeded, i, j);
                substitutionForLGPUStep1 << <1, maxNThreadsPerBlock, 0, gpuStreams[streamIndex] >> > (compactMatrix_, matrixY_, n, i, j);
                streamIndex = incrementStreamIndex(streamIndex);
                gpuErrchk(cudaPeekAtLastError());
                nThreadsNeeded -= maxNThreadsPerBlock;
                j += maxNThreadsPerBlock;
            }
            else
            {
                // printf("else: nThreadsNeeded: %d, i: %d, j:%d\n", nThreadsNeeded, i, j);
                substitutionForLGPUStep1 << <1, nThreadsNeeded, 0, gpuStreams[streamIndex] >> > (compactMatrix_, matrixY_, n, i, j);
                streamIndex = incrementStreamIndex(streamIndex);
                gpuErrchk(cudaPeekAtLastError());
                nThreadsNeeded = 0;
                j = i;
            }
        }
        cudaDeviceSynchronize();

        uint32_t nMergresNeeded = i / maxNThreadsPerBlock;
        if (nMergresNeeded > 0 && (i % maxNThreadsPerBlock != 0))
        {
            nMergresNeeded++;
        }
        nThreadsNeeded = nMergresNeeded / 2;
        if (nThreadsNeeded > 0)
        {
            // printf("step2: nThreadsNeeded: %d, i: %d, nMergresNeeded: %d\n", nThreadsNeeded, i, nMergresNeeded);
            substitutionForLGPUStep2 << <1, nThreadsNeeded, 0, gpuStreams[streamIndex] >> > (compactMatrix_, n, i, nMergresNeeded, maxNThreadsPerBlock);
            streamIndex = incrementStreamIndex(streamIndex);
            gpuErrchk(cudaPeekAtLastError());
        }
        cudaDeviceSynchronize();

        if (i == 0)
        {
            // printf("step3:%d\n");
            substitutionForLGPUStep3 << <1, 1, 0, gpuStreams[streamIndex] >> > (matrixB_, matrixY_, i);
            streamIndex = incrementStreamIndex(streamIndex);
            gpuErrchk(cudaPeekAtLastError());
        }
        else
        {
            // printf("step4:%d\n");
            substitutionForLGPUStep4 << <1, 1, 0, gpuStreams[streamIndex] >> > (compactMatrix_, matrixB_, matrixY_, n, i);
            streamIndex = incrementStreamIndex(streamIndex);
            gpuErrchk(cudaPeekAtLastError());
        }
        cudaDeviceSynchronize();
    }
}
////////// END: L Substitution //////////











////////// U Substitution //////////
__global__ void substitutionForUGPUStep1(float compactMatrix_[], float matrixX_[], uint32_t n, uint32_t i, uint32_t jStart)
{
    uint32_t myIndexInCompactMatrix = i * n + jStart + threadIdx.x;

    compactMatrix_[myIndexInCompactMatrix] = (compactMatrix_[myIndexInCompactMatrix] * matrixX_[threadIdx.x + jStart]);
    __syncthreads();

    for (uint32_t nReductionElements = blockDim.x; nReductionElements > 1; nReductionElements /= 2)
    {
        if (nReductionElements % 2 != 0)
        {
            if (threadIdx.x == nReductionElements - 3) // we ask second last thread to do this so that we do not have to sync
            {
                compactMatrix_[myIndexInCompactMatrix] += compactMatrix_[myIndexInCompactMatrix + 2];
            }
            nReductionElements--;
        }

        if (threadIdx.x < nReductionElements && threadIdx.x % 2 == 0)
        {
            float result = compactMatrix_[myIndexInCompactMatrix] + compactMatrix_[myIndexInCompactMatrix + 1];
            __syncthreads();
            compactMatrix_[i * n + jStart + (threadIdx.x / 2)] = result;
            __syncthreads();
        }
    }
}




__global__ void substitutionForUGPUStep2(float compactMatrix_[], uint32_t n, uint32_t i, uint32_t nMergresNeeded, uint32_t distanceBetweenElements)
{
    uint32_t myIndexInCompactMatrix = i * n + (i + 1) + (distanceBetweenElements * 2 * threadIdx.x); // (i + 1) to offset correctly after i column

    if (nMergresNeeded % 2 != 0)
    {
        if (threadIdx.x == blockDim.x - 1) // we ask second last thread to do this so that we do not have to sync
        {
            compactMatrix_[myIndexInCompactMatrix] += compactMatrix_[myIndexInCompactMatrix + (distanceBetweenElements * 2)];
        }
        nMergresNeeded--;
    }

    float result = compactMatrix_[myIndexInCompactMatrix] + compactMatrix_[myIndexInCompactMatrix + distanceBetweenElements];
    __syncthreads();
    compactMatrix_[i * n + (i + 1) + threadIdx.x] = result;
    __syncthreads();
    myIndexInCompactMatrix = i * n + (i + 1) + threadIdx.x;


    for (uint32_t nReductionElements = blockDim.x; nReductionElements > 1; nReductionElements /= 2)
    {
        if (nReductionElements % 2 != 0)
        {
            if (threadIdx.x == nReductionElements - 3) // we ask second last thread to do this so that we do not have to sync
            {
                compactMatrix_[myIndexInCompactMatrix] += compactMatrix_[myIndexInCompactMatrix + 2];
            }
            nReductionElements--;
        }

        if (threadIdx.x < nReductionElements && threadIdx.x % 2 == 0)
        {
            float result = compactMatrix_[myIndexInCompactMatrix] + compactMatrix_[myIndexInCompactMatrix + 1];
            __syncthreads();
            compactMatrix_[i * n + (i + 1) + (threadIdx.x / 2)] = result;
            __syncthreads();
        }
    }
}


__global__ void substitutionForUGPUStep3(float compactMatrix_[], float matrixY_[], float matrixX_[], uint32_t n, uint32_t i)
{
    uint32_t myIndexInCompactMatrix = i * n + i;
    matrixX_[i] = matrixY_[i] / compactMatrix_[myIndexInCompactMatrix];
}



__global__ void substitutionForUGPUStep4(float compactMatrix_[], float matrixY_[], float matrixX_[], uint32_t n, uint32_t i)
{
    uint32_t myIndexInCompactMatrix = i * n + i;
    matrixX_[i] = (matrixY_[i] - compactMatrix_[myIndexInCompactMatrix + 1]) / compactMatrix_[myIndexInCompactMatrix];
}



// AX = B
// A = LU
// LUX = B
// We can solve that by solving the 2 below
// 1) LY = B
// 2) UX = Y
// In this function solving for 2)
// size of compactMatrix is nXn while y and X are nX1, X is unkown
void substitutionForU(float compactMatrix_[], float matrixY_[], float matrixX_[], const uint32_t n, uint32_t maxNThreadsPerBlock)
{
    for (int64_t i = n - 1; i > -1; i--)
    {
        uint32_t nThreadsNeeded = (n - 1) - i;
        uint32_t streamIndex = 0;
        for (uint32_t j = i + 1; j < n;)
        {
            if (nThreadsNeeded >= maxNThreadsPerBlock) // 1024 because it is maximum thread allowed per Multiprocessor
            {
                // printf("if: nThreadsNeeded: %d, i: %d, j:%d.\n", nThreadsNeeded, i, j);
                substitutionForUGPUStep1 << <1, maxNThreadsPerBlock, 0, gpuStreams[streamIndex] >> > (compactMatrix_, matrixX_, n, i, j);
                streamIndex = incrementStreamIndex(streamIndex);
                gpuErrchk(cudaPeekAtLastError());
                nThreadsNeeded -= maxNThreadsPerBlock;
                j += maxNThreadsPerBlock;
            }
            else
            {
                // printf("else: nThreadsNeeded: %d, i: %d, j:%d\n", nThreadsNeeded, i, j);
                substitutionForUGPUStep1 << <1, nThreadsNeeded, 0, gpuStreams[streamIndex] >> > (compactMatrix_, matrixX_, n, i, j);
                streamIndex = incrementStreamIndex(streamIndex);
                gpuErrchk(cudaPeekAtLastError());
                nThreadsNeeded = 0;
                j = n;
            }
        }
        cudaDeviceSynchronize();

        uint32_t nSolvedElements = n - (i + 1);
        uint32_t nMergresNeeded = nSolvedElements / maxNThreadsPerBlock;
        if (nMergresNeeded > 0 && (nSolvedElements % maxNThreadsPerBlock != 0))
        {
            nMergresNeeded++;
        }
        nThreadsNeeded = nMergresNeeded / 2;
        if (nThreadsNeeded > 0)
        {
            // printf("step2: nThreadsNeeded: %d, i: %d, nMergresNeeded: %d\n", nThreadsNeeded, i, nMergresNeeded);
            substitutionForUGPUStep2 << <1, nThreadsNeeded, 0, gpuStreams[streamIndex] >> > (compactMatrix_, n, i, nMergresNeeded, maxNThreadsPerBlock);
            streamIndex = incrementStreamIndex(streamIndex);
            gpuErrchk(cudaPeekAtLastError());
        }
        cudaDeviceSynchronize();

        if (i == n - 1)
        {
            // printf("step3:%d\n");
            substitutionForUGPUStep3 << <1, 1, 0, gpuStreams[streamIndex] >> > (compactMatrix_, matrixY_, matrixX_, n, i);
            streamIndex = incrementStreamIndex(streamIndex);
            gpuErrchk(cudaPeekAtLastError());
        }
        else
        {
            // printf("step4:%d\n");
            substitutionForUGPUStep4 << <1, 1, 0, gpuStreams[streamIndex] >> > (compactMatrix_, matrixY_, matrixX_, n, i);
            streamIndex = incrementStreamIndex(streamIndex);
            gpuErrchk(cudaPeekAtLastError());
        }
        cudaDeviceSynchronize();
    }
}
////////// END: U Substitution //////////











////////// GPU Multiply Special Case //////////
__global__ void multiplyGpuStep1(float matrixKTranspose_[], float matrixX_[], uint32_t n, uint32_t iStart)
{
    uint32_t myIndex = iStart + threadIdx.x;

    matrixX_[myIndex] = matrixX_[myIndex] * matrixKTranspose_[myIndex];
    __syncthreads();

    for (uint32_t nReductionElements = blockDim.x; nReductionElements > 1; nReductionElements /= 2)
    {
        if (nReductionElements % 2 != 0)
        {
            if (threadIdx.x == nReductionElements - 3) // we ask second last thread to do this so that we do not have to sync
            {
                matrixX_[myIndex] += matrixX_[myIndex + 2];
            }
            nReductionElements--;
        }

        if (threadIdx.x < nReductionElements && threadIdx.x % 2 == 0)
        {
            float result = matrixX_[myIndex] + matrixX_[myIndex + 1];
            __syncthreads();
            matrixX_[iStart + (threadIdx.x / 2)] = result;
            __syncthreads();
        }
    }
}




__global__ void multiplyGpuStep2(float matrixX_[], uint32_t nMergresNeeded, uint32_t distanceBetweenElements)
{
    uint32_t myIndex = (distanceBetweenElements * 2 * threadIdx.x);


    // printf("tId: %d, myIndex: %d, distanceBetweenElements: %d, nMergresNeeded:%d.\n", threadIdx.x, myIndex, distanceBetweenElements, nMergresNeeded);

    if (nMergresNeeded % 2 != 0) // nMergresNeeded
    {
        // nMergresNeeded = 1 does not need.
        if (threadIdx.x == blockDim.x - 1) // we ask second last thread to do this so that we do not have to sync
        {
            // printf("Odd1: tId: %d, myIndex: %d, distanceBetweenElements: %d, nMergresNeeded:%d.\n", threadIdx.x, myIndex, distanceBetweenElements, nMergresNeeded);
            matrixX_[myIndex] += matrixX_[myIndex + (distanceBetweenElements * 2)];
        }
        nMergresNeeded--;
    }

    // printf("tId: %d, myIndex: %d, distanceBetweenElements: %d, nMergresNeeded:%d.\n", threadIdx.x, myIndex, distanceBetweenElements, nMergresNeeded);
    float result = matrixX_[myIndex] + matrixX_[myIndex + distanceBetweenElements];
    __syncthreads();
    matrixX_[threadIdx.x] = result;
    __syncthreads();
    myIndex = threadIdx.x;


    for (uint32_t nReductionElements = blockDim.x; nReductionElements > 1; nReductionElements /= 2)
    {
        if (nReductionElements % 2 != 0)
        {
            if (threadIdx.x == nReductionElements - 3) // we ask second last thread to do this so that we do not have to sync
            {
                // printf("Odd2: tId: %d, myIndex: %d, distanceBetweenElements: %d, nMergresNeeded:%d.\n", threadIdx.x, myIndex, distanceBetweenElements, nMergresNeeded);
                matrixX_[myIndex] += matrixX_[myIndex + 2];
            }
            nReductionElements--;
        }

        if (threadIdx.x < nReductionElements && threadIdx.x % 2 == 0)
        {
            float result = matrixX_[myIndex] + matrixX_[myIndex + 1];
            __syncthreads();
            matrixX_[(threadIdx.x / 2)] = result;
            __syncthreads();
        }
    }
}



__global__ void multiplyGpuStep3(float matrixX_[], float result_[])
{
    result_[0] = matrixX_[0];
}


void multiplyGpu(float matrixKTranspose_[], float matrixX_[], float matrixResult_[], const uint32_t n, uint32_t maxNThreadsPerBlock)
{
    uint32_t nThreadsNeeded = n;
    uint32_t streamIndex = 0;
    for (uint32_t i = 0; i < n;)
    {
        if (nThreadsNeeded >= maxNThreadsPerBlock) // 1024 because it is maximum thread allowed per Multiprocessor
        {
            // printf("if: nThreadsNeeded: %d, i: %d.\n", nThreadsNeeded, i);
            multiplyGpuStep1 << <1, maxNThreadsPerBlock, 0, gpuStreams[streamIndex] >> > (matrixKTranspose_, matrixX_, n, i);
            streamIndex = incrementStreamIndex(streamIndex);
            gpuErrchk(cudaPeekAtLastError());
            nThreadsNeeded -= maxNThreadsPerBlock;
            i += maxNThreadsPerBlock;
        }
        else
        {
            // printf("else: nThreadsNeeded: %d, i: %d\n", nThreadsNeeded, i);
            multiplyGpuStep1 << <1, nThreadsNeeded, 0, gpuStreams[streamIndex] >> > (matrixKTranspose_, matrixX_, n, i);
            streamIndex = incrementStreamIndex(streamIndex);
            gpuErrchk(cudaPeekAtLastError());
            nThreadsNeeded = 0;
            i = n;
        }
    }

    cudaDeviceSynchronize();

    uint32_t nMergresNeeded = n / maxNThreadsPerBlock;
    if (nMergresNeeded > 0 && (n % maxNThreadsPerBlock != 0))
    {
        nMergresNeeded++;
    }
    nThreadsNeeded = nMergresNeeded / 2;
    if (nThreadsNeeded > 0)
    {
        // printf("step2: nThreadsNeeded: %d, nMergresNeeded: %d\n", nThreadsNeeded, nMergresNeeded);
        multiplyGpuStep2 << <1, nThreadsNeeded, 0, gpuStreams[streamIndex] >> > (matrixX_, nMergresNeeded, maxNThreadsPerBlock);
        streamIndex = incrementStreamIndex(streamIndex);
        gpuErrchk(cudaPeekAtLastError());
    }

    cudaDeviceSynchronize();

    multiplyGpuStep3 << <1, 1, 0, gpuStreams[streamIndex] >> > (matrixX_, matrixResult_);
}
////////// EMD: GPU Multiply Special Case //////////









int main(int argc, char* argv[])
{
    N_THREADS_PER_BLOCK = 256; // always less than 1024
    SHARED_MEMORY_SIZE = 25; // SHARED_MEMORY_SIZE = N_THREADS_PER_BLOCK * N, N >= 1 and integer
    N_STREAM = 1;
    

    float predictionPointX, predictionPointY;
    uint32_t gridM;

    if (argc < 4)
    {
        printf("Not enough arguments were provided!!! Make sure you add grid size and your points x and i");
        return 0;
    }
    else
    {
        gridM = std::stol(argv[1], nullptr, 10);
        predictionPointX = std::stod(argv[2]);
        predictionPointY = std::stod(argv[3]);

        if (argc == 7)
        {
            N_THREADS_PER_BLOCK = std::stol(argv[4], nullptr, 10);
            SHARED_MEMORY_SIZE = std::stol(argv[5], nullptr, 10);
            N_STREAM = std::stol(argv[6], nullptr, 10);
        }
    }


    printf("nT/B: %d, Shared M: %d, nStream: %d.\n", N_THREADS_PER_BLOCK, SHARED_MEMORY_SIZE, N_STREAM);

    gpuStreams = new cudaStream_t[N_STREAM];

#ifdef PRINT_MATRIX
    print_device_properties();
#endif

    cudaSetDevice(0);

    //struct timeval start, stop;
    //double total_time;


    float gridH = 1.0 / (1.0 + gridM);


    float* gridPosition = new float[gridM];
    for (uint32_t i = 0; i < gridM; i++)
    {
        gridPosition[i] = gridH * (i + 1);
    }


    float* gridPositionSubtractedFromHalfAndSquared = new float[gridM];
    for (uint32_t i = 0; i < gridM; i++)
    {
        gridPositionSubtractedFromHalfAndSquared[i] = gridPosition[i] - 0.5;
        gridPositionSubtractedFromHalfAndSquared[i] = gridPositionSubtractedFromHalfAndSquared[i] * gridPositionSubtractedFromHalfAndSquared[i];
    }

    float* observedValues = new float[gridM * gridM];
    for (uint32_t i = 0; i < gridM; i++)
    {
        for (uint32_t j = i; j < gridM; j++)
        {
            float withoutNoise = 1 - (gridPositionSubtractedFromHalfAndSquared[i] + gridPositionSubtractedFromHalfAndSquared[j]);
            observedValues[i * gridM + j] = withoutNoise + get5HundredthsRandom();
            observedValues[j * gridM + i] = withoutNoise + get5HundredthsRandom();
        }
    }

#ifdef PRINT_MATRIX
    //printf("gridPosition:\n");
    //printMatrix(gridPosition, gridM, 1);
    //printf("observedValues:\n");
    //printMatrix(observedValues, gridM, gridM);
#endif


    uint32_t n = gridM * gridM;
    float gpuTime;
    cudaEvent_t startGpu, stopGpu;
    cudaEventCreate(&startGpu);
    cudaEventCreate(&stopGpu);
    cudaError_t errorCode;
    float* gridPositionGpu;
    size_t gridPositionGpuSize = gridM * sizeof(float);
    float* observedValuesGpu;
    size_t observedValuesGpuSize = gridM * gridM * sizeof(float);
    float* matrixKCaptiaGpu;
    size_t matrixKCaptiaGpuSize = n * n * sizeof(float);
    float* matrixKTransposeGpu;
    size_t matrixKTransposeGpuSize = gridM * gridM * sizeof(float);
    float* matrixYGpu;
    size_t matrixYGpuSize = n * 1 * sizeof(float);
    float* matrixXGpu;
    size_t matrixXGpuSize = n * 1 * sizeof(float);
    float* resultGpu;
    size_t resultGpuSize = sizeof(float);
    float resultCpu[1];
    float* print;


    //printf("GPU Begins:\n");
    //printf("Step1:  create %d streams:\n", N_STREAM);
    for (int i = 0; i < N_STREAM; i++)
    {
        gpuErrchk(cudaStreamCreate(&gpuStreams[i]));
    }

    cudaEventRecord(startGpu, 0);
    // printf("Step2:  cudaMalloc:\n");
    errorCode = cudaMalloc(&gridPositionGpu, gridPositionGpuSize);
    gpuErrchk(errorCode);
    errorCode = cudaMalloc(&observedValuesGpu, observedValuesGpuSize);
    gpuErrchk(errorCode);
    errorCode = cudaMalloc(&matrixKCaptiaGpu, matrixKCaptiaGpuSize);
    gpuErrchk(errorCode);
    errorCode = cudaMalloc(&matrixKTransposeGpu, matrixKTransposeGpuSize);
    gpuErrchk(errorCode);
    errorCode = cudaMalloc(&matrixYGpu, matrixYGpuSize);
    gpuErrchk(errorCode);
    errorCode = cudaMalloc(&matrixXGpu, matrixXGpuSize);
    gpuErrchk(errorCode);
    errorCode = cudaMalloc(&resultGpu, resultGpuSize);
    gpuErrchk(errorCode);


    // printf("Step3:  Coppied to GPU:\n");
    errorCode = cudaMemcpy(gridPositionGpu, gridPosition, gridPositionGpuSize, cudaMemcpyHostToDevice);
    gpuErrchk(errorCode);
    errorCode = cudaMemcpy(observedValuesGpu, observedValues, observedValuesGpuSize, cudaMemcpyHostToDevice);


#ifdef PRINT_MATRIX
    print = new float[gridM * 1];
    cudaMemcpy(print, gridPositionGpu, gridPositionGpuSize, cudaMemcpyDeviceToHost);
    gpuErrchk(errorCode);
    printf("gridPositionGpu:\n");
    printMatrix(print, gridM, 1);

    print = new float[gridM * gridM];
    cudaMemcpy(print, observedValuesGpu, observedValuesGpuSize, cudaMemcpyDeviceToHost);
    gpuErrchk(errorCode);
    printf("observedValuesGpu:\n");
    printMatrix(print, n, 1);
#endif

    // printf("Step4: generate K matrix size %d X %d, nStreams: %d\n", n, n, N_STREAM);
    generateCaptialKOnGpu(matrixKCaptiaGpu, gridPositionGpu, gridM);
    cudaDeviceSynchronize(); // important for the next step

#ifdef PRINT_MATRIX
    print = new float[n * n];
    cudaMemcpy(print, matrixKCaptiaGpu, matrixKCaptiaGpuSize, cudaMemcpyDeviceToHost);
    gpuErrchk(errorCode);
    printf("matrixKCaptiaGpu:\n");
    printMatrix(print, n, n);
#endif

    // printf("Step5: Add noise to matrix K soze: %d X %d, nStreams: %d, nThreads/block: %d\n", n, n, N_STREAM, N_THREADS_PER_BLOCK);
    addNoiseOnGpu(matrixKCaptiaGpu, gridM, N_THREADS_PER_BLOCK);

#ifdef PRINT_MATRIX
    print = new float[n * n];
    cudaMemcpy(print, matrixKCaptiaGpu, matrixKCaptiaGpuSize, cudaMemcpyDeviceToHost);
    gpuErrchk(errorCode);
    printf("matrixKCaptiaGpu:\n");
    printMatrix(print, n, n);
#endif

    // printf("Step6: k transpose Matrix of %d X %d being computated, nStreams: %d\n", gridM, gridM, N_STREAM);
    generateKTransposeOnGpu(matrixKTransposeGpu, gridPositionGpu, gridM, predictionPointX, predictionPointY);
    cudaDeviceSynchronize(); // important for the next step

#ifdef PRINT_MATRIX
    print = new float[n * n];
    cudaMemcpy(print, matrixKTransposeGpu, matrixKTransposeGpuSize, cudaMemcpyDeviceToHost);
    gpuErrchk(errorCode);
    printf("matrixKTransposeGpu:\n");
    printMatrix(print, n, 1);
#endif

    // printf("Step7: Compute LU Matrix of %d X %d, nStreams: %d, nThreads/block: %d, shared memory: %d floats.\n", n, n, N_STREAM, N_THREADS_PER_BLOCK, SHARED_MEMORY_SIZE);
    computeLUMatrixGpuCompact(matrixKCaptiaGpu, n, N_THREADS_PER_BLOCK);// sync inside function

#ifdef PRINT_MATRIX
    print = new float[n * n];
    cudaMemcpy(print, matrixKCaptiaGpu, matrixKCaptiaGpuSize, cudaMemcpyDeviceToHost);
    gpuErrchk(errorCode);
    printf("matrixKCaptiaGpu:\n");
    printMatrix(print, n, n);
#endif

    // printf("Step8: Substitution for L Matrix of %d X %d, nStreams: %d, nThreads/block: %d\n", n, n, N_STREAM, N_THREADS_PER_BLOCK);
    substitutionForL(matrixKCaptiaGpu, observedValuesGpu, matrixYGpu, n, N_THREADS_PER_BLOCK);  // sync inside function

#ifdef PRINT_MATRIX
    print = new float[n * 1];
    cudaMemcpy(print, matrixYGpu, matrixYGpuSize, cudaMemcpyDeviceToHost);
    gpuErrchk(errorCode);
    printf("matrixYGpu:\n");
    printMatrix(print, n, 1);
#endif

    //printf("Step9: Substitution for U Matrix of %d X %d, nStreams: %d, nThreads/block: %d\n", n, n, N_STREAM, N_THREADS_PER_BLOCK);
    substitutionForU(matrixKCaptiaGpu, matrixYGpu, matrixXGpu, n, N_THREADS_PER_BLOCK); // sync inside function

#ifdef PRINT_MATRIX
    print = new float[n * 1];
    cudaMemcpy(print, matrixXGpu, matrixXGpuSize, cudaMemcpyDeviceToHost);
    gpuErrchk(errorCode);
    printf("matrixXGpu:\n");
    printMatrix(print, n, 1);
#endif

    // printf("Step10: Final Multiplication for Matrix of %d X %d, nStreams: %d, nThreads/block: %d\n", n, n, N_STREAM, N_THREADS_PER_BLOCK);
    multiplyGpu(matrixKTransposeGpu, matrixXGpu, resultGpu, n, N_THREADS_PER_BLOCK); // sync inside function


    cudaMemcpy(resultCpu, resultGpu, resultGpuSize, cudaMemcpyDeviceToHost);
    gpuErrchk(errorCode);
    // printf("Step11: Coppied resultCpu to Device. result is: %f\n", resultCpu[0]);

    cudaEventRecord(stopGpu, 0);
    cudaEventSynchronize(stopGpu);
    cudaEventElapsedTime(&gpuTime, startGpu, stopGpu);
    printf("Total time = %lf seconds, Predicted Value = %lf\n", gpuTime/1000, resultCpu[0]);

    gpuErrchk(cudaFree(gridPositionGpu));
    gpuErrchk(cudaFree(observedValuesGpu));
    gpuErrchk(cudaFree(matrixKCaptiaGpu));
    gpuErrchk(cudaFree(matrixKTransposeGpu));
    gpuErrchk(cudaFree(matrixYGpu));
    gpuErrchk(cudaFree(matrixXGpu));
    gpuErrchk(cudaFree(resultGpu));


    for (int i = 0; i < N_STREAM; i++)
    {
        gpuErrchk(cudaStreamDestroy(gpuStreams[i]));
    }
    // printf("GPU has released %d streams\n", N_STREAM);
}