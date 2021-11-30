// LUStage 1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <cuda_runtime.h>

// #include <chrono>
#include <time.h>       /* time */
#include <sys/time.h>

#include <iostream>
#include <algorithm>


#define COMPLETE_TEST // comment to only test timing
const uint32_t N_THREADS_PER_BLOCK = 256; // always less than 1024
const uint32_t SHARED_MEMORY_SIZE = 25; // SHARED_MEMORY_SIZE = N_THREADS_PER_BLOCK * N, N >= 1 and integer
const uint32_t N_STREAM = 35;
cudaStream_t gpuStreams[N_STREAM];



const uint64_t MY_MATRIX_N = 5000;
float myMatrix[MY_MATRIX_N * MY_MATRIX_N] = { 0 };

float matrixL[MY_MATRIX_N * MY_MATRIX_N] = { 0 };
float matrixL2[MY_MATRIX_N * MY_MATRIX_N] = { 0 };


float matrixU[MY_MATRIX_N * MY_MATRIX_N] = { 0 };
float matrixU2[MY_MATRIX_N * MY_MATRIX_N] = { 0 };

float matrixLU[MY_MATRIX_N * MY_MATRIX_N] = { 0 };
float inputOutputResult[MY_MATRIX_N * MY_MATRIX_N] = { 0 };



#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    }
}



void printSquareMatrix(const float matrix[], const uint64_t n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            //printf("  %lf", matrix[j + i * n]);
            printf("  %f", matrix[j + i * n]);
        printf("\n");
    }
}


void generateSquareMatrix(float matrix[], const uint64_t n)
{
    uint64_t index = 0;
    for (uint64_t i = 0; i < n; i++)
    {
        for (uint64_t j = 0; j < n; j++)
        {
            matrix[index] = (float)rand() / (float)RAND_MAX;
            index++;
        }
    }
}


void duplicateSquareMatrix(const float matrixSource[], float matrixDestination[], const uint64_t n)
{
    std::copy(&matrixSource[0], &matrixSource[n * n], &matrixDestination[0]);
}


void fillSquareMatrix(const float matrixSource[], const uint64_t n, const float value)
{
    // std::fill(&matrixSource[0], &matrixSource[n * n], value);
}


bool compareMatrix1Digit(const float matrix1[], const float matrix2[], const uint64_t n)
{
    uint64_t index = 0;
    for (uint64_t i = 0; i < n; i++)
    {
        for (uint64_t j = 0; j < n; j++)
        {
            float a = fabsf(matrix1[index]);
            float delta = matrix1[index] - matrix2[index];
            if(a > 1)
                delta /= matrix1[index];
            if (delta < 0)
                delta = -delta;
            if (delta > 0.3)
            {
                // printf("index = %d, %f vs %f, error %f, delta %f.\n", index, matrix1[index], matrix2[index], matrix1[index] - matrix2[index], delta);
                return false;
            }
            index++;
        }
    }
    return true;
}



struct MatrixDataForMultiplication
{
    // startingI: starting I of the interested area
    // startingJ: starting J of the interested area
    // iSize: the number of rows for area of interest
    // jSize: the number of columns  for area of interest
    // matrixJSize: this is the main actual matrix N number of columns, 
    // since I am not using a 2-dimensional array this is needed to make sure I index correctly.
    uint64_t startingI, startingJ, iSize, jSize, matrixJSize;
};


void multiplyMatrixSerial(const float matrixA[], const MatrixDataForMultiplication& matrixAData,
    const float matrixB[], const MatrixDataForMultiplication& matrixBData,
    float result[], const MatrixDataForMultiplication& resultData, bool isPositive)
{
    for (uint64_t i = 0; i < matrixAData.iSize; i++)
    {
        for (uint64_t j = 0; j < matrixBData.jSize; j++)
        {
            uint64_t matrixAIndex = matrixAData.startingJ + (matrixAData.startingI + i) * matrixAData.matrixJSize;
            uint64_t matrixBIndex = matrixBData.startingJ + j + matrixBData.startingI * matrixBData.matrixJSize;
            uint64_t resultIndex = resultData.startingJ + j + (resultData.startingI + i) * resultData.matrixJSize;
            result[resultIndex] = 0;
            // printf("[%lu, %lu] =>", i, j);
            for (uint64_t k = 0; k < matrixAData.jSize; k++)
            {
                // printf("%.2lf X %.2lf + ", matrixA[matrixAIndex], matrixB[matrixBIndex]);
                result[resultIndex] += matrixA[matrixAIndex] * matrixB[matrixBIndex];
                matrixAIndex++;
                matrixBIndex += matrixBData.matrixJSize;
            }

            if (!isPositive)
            {
                result[resultIndex] = -result[resultIndex];
            }
            // printf("= %.2lf \n", result[resultIndex]);
        }
    }
}




void computeLUMatrix(float matrix[], float matrixL[], float matrixU[], const uint64_t n)
{
    std::copy(&matrix[0], &matrix[n], &matrixU[0]);

    for (uint64_t i = 0; i < n; i++)
    {
        matrixL[i * n] = matrix[i * n];
        matrixL[i * n + i] = 1;
    }

    for (uint64_t k = 0; k < n - 1; k++)
    {
        for (uint64_t i = k + 1; i < n; i++)
        {
            float toBeL = matrix[i * n + k] / matrix[k * n + k];
            for (uint64_t j = k + 1; j < n; j++)
            {
                matrix[i * n + j] = matrix[i * n + j] - toBeL * matrix[k * n + j];
            }

            for (uint64_t j = i; j < n; j++)
            {
                matrixU[(k + 1) * n + j] = matrix[(k + 1) * n + j];
            }

            matrixL[i * n + k] = toBeL;
        }
    }
}



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


uint32_t incrementStreamIndex(uint32_t streamIndex)
{
    streamIndex++;
    if (streamIndex >= N_STREAM)
    {
        streamIndex = 0;
    }
    return streamIndex;
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




void extractLUMartixFromCompactResult(float matrix[], float matrixL[], float matrixU[], const uint64_t n)
{
    for (uint64_t i = 0; i < n; i++)
    {
        for (uint64_t j = 0; j < i; j++)
        {
            matrixU[i * n + j] = 0;
        }
        for (uint64_t j = i; j < n; j++)
        {
            matrixU[i * n + j] = matrix[i * n + j];
        }
    }

    for (int64_t j = 0; j < n; j++)
    {
        for (int64_t i = 0; i < j; i++)
        {
            matrixL[i * n + j] = 0;
        }
        matrixL[j * n + j] = 1;
        for (int64_t i = j + 1; i < n; i++)
        {
            matrixL[i * n + j] = matrix[i * n + j];
        }
    }
}



void setSquareMatrixTo(const float input, float matrix[], const uint64_t n)
{
    for (uint64_t i = 0; i < n; i++)
    {
        for (uint64_t j = 0; j < n; j++)
        {
            matrix[i * n + j] = 0;
        }
    }
}



// ---------------------------------------------------------------------------- 
// Print device properties
void print_device_properties(uint32_t& maxNThreadsPerBlock) {
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

    maxNThreadsPerBlock = deviceProp.maxThreadsPerBlock;
}
// ---------------------------------------------------------------------------- 
// Main program - initializes points and computes minimum distance 
// between the points
//







int main()
{
    float gpuTime;
    cudaEvent_t startGpu, stopGpu;
    cudaEventCreate(&startGpu);
    cudaEventCreate(&stopGpu);
    // std::chrono::high_resolution_clock::time_point startTimeCpu, endTimeCpu;
    struct timeval start, stop;
    double total_time;
    cudaError_t errorCode;

    uint32_t testSize;
    size_t myTestMatrixSize;
    float* gpuMatrix;
    uint32_t nTests;

    uint32_t maxNThreadsPerBlock;
    print_device_properties(maxNThreadsPerBlock);
    cudaSetDevice(0);


    for (int i = 0; i < N_STREAM; i++)
    {
        gpuErrchk(cudaStreamCreate(&gpuStreams[i]));
    }
    printf("GPU has created %d streams\n", N_STREAM);




    #ifdef COMPLETE_TEST
    generateSquareMatrix(myMatrix, MY_MATRIX_N);


    // test block for correctness and time
    // initiak setup for each test
    testSize = 200;
    printf("\n\nComplete test for Matrix size %d X %d\n", testSize, testSize);
    myTestMatrixSize = testSize * testSize * sizeof(float);
    gpuMatrix;
    errorCode = cudaMalloc(&gpuMatrix, myTestMatrixSize);
    gpuErrchk(errorCode);

    // copy data
    duplicateSquareMatrix(myMatrix, inputOutputResult, testSize);

    // CPU
    printf("CPU Begins:\n");
    // startTimeCpu = std::chrono::high_resolution_clock::now();
    gettimeofday(&start, NULL); 
    computeLUMatrix(inputOutputResult, matrixL, matrixU, testSize);
    // endTimeCpu = std::chrono::high_resolution_clock::now();
    gettimeofday(&stop, NULL); 
    // auto deltaCpuTime = endTimeCpu - startTimeCpu;
    total_time = (stop.tv_sec-start.tv_sec)+0.000001*(stop.tv_usec-start.tv_usec);
    printf("Computing LU a %d X %d matrix by CPU took: ", testSize, testSize);
    // std::cout << deltaCpuTime / std::chrono::milliseconds(1) << " ms\n";
    std::cout << 1000 * total_time << " ms\n";


    // GPU
    printf("GPU Begins:\n");
    cudaEventRecord(startGpu, 0);
    errorCode = cudaMemcpy(gpuMatrix, myMatrix, myTestMatrixSize, cudaMemcpyHostToDevice);
    gpuErrchk(errorCode);
    cudaEventRecord(stopGpu, 0);
    cudaEventSynchronize(stopGpu);
    cudaEventElapsedTime(&gpuTime, startGpu, stopGpu);
    printf("1) Copying a %d X %d matrix (# bytes: %d) from RAM to GPU: took %fms\n", testSize, testSize, myTestMatrixSize, gpuTime);

    cudaEventRecord(startGpu, 0);
    computeLUMatrixGpuCompact(gpuMatrix, testSize, N_THREADS_PER_BLOCK);
    cudaEventRecord(stopGpu, 0);
    cudaEventSynchronize(stopGpu);
    cudaEventElapsedTime(&gpuTime, startGpu, stopGpu);
    printf("2) Computing LU for %d X %d matrix by #thread/block: %d and Shared Memory Size: %d float: GPU took %fms\n", testSize, testSize, N_THREADS_PER_BLOCK, SHARED_MEMORY_SIZE, gpuTime);

    cudaEventRecord(startGpu, 0);
    cudaMemcpy(inputOutputResult, gpuMatrix, myTestMatrixSize, cudaMemcpyDeviceToHost);
    cudaEventRecord(stopGpu, 0);
    cudaEventSynchronize(stopGpu);
    cudaEventElapsedTime(&gpuTime, startGpu, stopGpu);
    printf("3) Copying a %d X %d matrix (# bytes: %d) from GPU to RAM: took %f ms\n", testSize, testSize, myTestMatrixSize, gpuTime);

    // comparing result
    printf("Comparing Result:\n");
    extractLUMartixFromCompactResult(inputOutputResult, matrixL2, matrixU2, testSize);
    printf("Lower Matrix Match? %s\n", compareMatrix1Digit(matrixL, matrixL2, testSize) ? "true" : "false");
    printf("Upper Matrix Match? %s\n", compareMatrix1Digit(matrixU, matrixU2, testSize) ? "true" : "false");

    setSquareMatrixTo(0.0, matrixL, testSize);
    setSquareMatrixTo(0.0, matrixU, testSize);

    // end of each test
    cudaFree(gpuMatrix);





    // test block for correctness and time
    // initiak setup for each test
    testSize = 1500;
    printf("\n\nComplete test for Matrix size %d X %d\n", testSize, testSize);
    myTestMatrixSize = testSize * testSize * sizeof(float);
    gpuMatrix;
    errorCode = cudaMalloc(&gpuMatrix, myTestMatrixSize);
    gpuErrchk(errorCode);

    // copy data
    duplicateSquareMatrix(myMatrix, inputOutputResult, testSize);

    // CPU
    printf("CPU Begins:\n");
    // startTimeCpu = std::chrono::high_resolution_clock::now();
    gettimeofday(&start, NULL); 
    computeLUMatrix(inputOutputResult, matrixL, matrixU, testSize);
    // endTimeCpu = std::chrono::high_resolution_clock::now();
    gettimeofday(&stop, NULL); 
    // auto deltaCpuTime = endTimeCpu - startTimeCpu;
    total_time = (stop.tv_sec-start.tv_sec)+0.000001*(stop.tv_usec-start.tv_usec);
    printf("Computing LU a %d X %d matrix by CPU took: ", testSize, testSize);
    // std::cout << deltaCpuTime / std::chrono::milliseconds(1) << " ms\n";
    std::cout << 1000 * total_time << " ms\n";


    // GPU
    printf("GPU Begins:\n");
    cudaEventRecord(startGpu, 0);
    errorCode = cudaMemcpy(gpuMatrix, myMatrix, myTestMatrixSize, cudaMemcpyHostToDevice);
    gpuErrchk(errorCode);
    cudaEventRecord(stopGpu, 0);
    cudaEventSynchronize(stopGpu);
    cudaEventElapsedTime(&gpuTime, startGpu, stopGpu);
    printf("1) Copying a %d X %d matrix (# bytes: %d) from RAM to GPU: took %fms\n", testSize, testSize, myTestMatrixSize, gpuTime);

    cudaEventRecord(startGpu, 0);
    computeLUMatrixGpuCompact(gpuMatrix, testSize, N_THREADS_PER_BLOCK);
    cudaEventRecord(stopGpu, 0);
    cudaEventSynchronize(stopGpu);
    cudaEventElapsedTime(&gpuTime, startGpu, stopGpu);
    printf("2) Computing LU for %d X %d matrix by #thread/block: %d and Shared Memory Size: %d float: GPU took %fms\n", testSize, testSize, N_THREADS_PER_BLOCK, SHARED_MEMORY_SIZE, gpuTime);

    cudaEventRecord(startGpu, 0);
    cudaMemcpy(inputOutputResult, gpuMatrix, myTestMatrixSize, cudaMemcpyDeviceToHost);
    cudaEventRecord(stopGpu, 0);
    cudaEventSynchronize(stopGpu);
    cudaEventElapsedTime(&gpuTime, startGpu, stopGpu);
    printf("3) Copying a %d X %d matrix (# bytes: %d) from GPU to RAM: took %f ms\n", testSize, testSize, myTestMatrixSize, gpuTime);

    // comparing result
    printf("Comparing Result:\n");
    extractLUMartixFromCompactResult(inputOutputResult, matrixL2, matrixU2, testSize);
    printf("Lower Matrix Match? %s\n", compareMatrix1Digit(matrixL, matrixL2, testSize) ? "true" : "false");
    printf("Upper Matrix Match? %s\n", compareMatrix1Digit(matrixU, matrixU2, testSize) ? "true" : "false");

    setSquareMatrixTo(0.0, matrixL, testSize);
    setSquareMatrixTo(0.0, matrixU, testSize);


    // end of each test
    cudaFree(gpuMatrix);
    #endif




    // GPU time testing
    // allocating maximum memory needed
    myTestMatrixSize = MY_MATRIX_N * MY_MATRIX_N * sizeof(float);
    cudaMalloc(&gpuMatrix, myTestMatrixSize);

    nTests = 3;
    testSize = 200;
    printf("\nTiming test for Matrix size %d X %d\n", testSize, testSize);
    for (uint32_t i = 0; i < nTests; i++)
    {
        cudaEventRecord(startGpu, 0);
        computeLUMatrixGpuCompact(gpuMatrix, testSize, N_THREADS_PER_BLOCK);
        cudaEventRecord(stopGpu, 0);
        cudaEventSynchronize(stopGpu);
        cudaEventElapsedTime(&gpuTime, startGpu, stopGpu);
        printf("Computing LU for %d X %d matrix by #thread/block: %d and Shared Memory Size: %d float: GPU took %fms\n", testSize, testSize, N_THREADS_PER_BLOCK, SHARED_MEMORY_SIZE, gpuTime);
    }


    nTests = 3;
    testSize = 500;
    printf("\nTiming test for Matrix size %d X %d\n", testSize, testSize);
    for (uint32_t i = 0; i < nTests; i++)
    {
        cudaEventRecord(startGpu, 0);
        computeLUMatrixGpuCompact(gpuMatrix, testSize, N_THREADS_PER_BLOCK);
        cudaEventRecord(stopGpu, 0);
        cudaEventSynchronize(stopGpu);
        cudaEventElapsedTime(&gpuTime, startGpu, stopGpu);
        printf("Computing LU for %d X %d matrix by #thread/block: %d and Shared Memory Size: %d float: GPU took %fms\n", testSize, testSize, N_THREADS_PER_BLOCK, SHARED_MEMORY_SIZE, gpuTime);
    }


    nTests = 3;
    testSize = 1000;
    printf("\nTiming test for Matrix size %d X %d\n", testSize, testSize);
    for (uint32_t i = 0; i < nTests; i++)
    {
        cudaEventRecord(startGpu, 0);
        computeLUMatrixGpuCompact(gpuMatrix, testSize, N_THREADS_PER_BLOCK);
        cudaEventRecord(stopGpu, 0);
        cudaEventSynchronize(stopGpu);
        cudaEventElapsedTime(&gpuTime, startGpu, stopGpu);
        printf("Computing LU for %d X %d matrix by #thread/block: %d and Shared Memory Size: %d float: GPU took %fms\n", testSize, testSize, N_THREADS_PER_BLOCK, SHARED_MEMORY_SIZE, gpuTime);
    }


    nTests = 3;
    testSize = 2000;
    printf("\nTiming test for Matrix size %d X %d\n", testSize, testSize);
    for (uint32_t i = 0; i < nTests; i++)
    {
        cudaEventRecord(startGpu, 0);
        computeLUMatrixGpuCompact(gpuMatrix, testSize, N_THREADS_PER_BLOCK);
        cudaEventRecord(stopGpu, 0);
        cudaEventSynchronize(stopGpu);
        cudaEventElapsedTime(&gpuTime, startGpu, stopGpu);
        printf("Computing LU for %d X %d matrix by #thread/block: %d and Shared Memory Size: %d float: GPU took %fms\n", testSize, testSize, N_THREADS_PER_BLOCK, SHARED_MEMORY_SIZE, gpuTime);
    }





    // free GPU resources
    gpuErrchk(cudaFree(gpuMatrix));
    for (int i = 0; i < N_STREAM; i++)
    {
        gpuErrchk(cudaStreamDestroy(gpuStreams[i]));
    }
    printf("GPU has released %d streams\n", N_STREAM);
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
