/*
nvcc -x cu -arch=sm_30 -std=c++11 -Xcompiler '-Wall,-Wextra' testCudacommon.cpp && ./a.out
*/

#include <algorithm>
#include <cassert>
#include <cstdlib>                  // rand
#include <iomanip>
#include <iostream>

#include "cudacommon.hpp"


__global__ void kernelTestSumEasy
(
    unsigned int         const              dpnToReduce   ,
    unsigned int const * const __restrict__ dpDataToReduce,
    unsigned int       * const __restrict__ dpResults
)
{
    if ( threadIdx.x == 0 )
    {
        unsigned int sum = 0;
        auto const n = dpnToReduce;
        for ( auto i = 0u; i < n; ++i )
            sum += dpDataToReduce[i];
        *dpResults = sum;
    }
}

__global__ void kernelTestSum
(
    unsigned int         const              dpnToReduce   ,
    unsigned int const * const __restrict__ dpDataToReduce,
    unsigned int       * const __restrict__ dpResults
)
{
    __shared__ int smBuffer[32];
    if ( threadIdx.x < dpnToReduce )
    {
        //auto const sum = blockReduceSum( (int)dpDataToReduce[ threadIdx.x ], smBuffer );
        auto const sum = warpReduceSum( (int)dpDataToReduce[ threadIdx.x ] );
        if ( threadIdx.x == 0 )
            *dpResults = sum;
    }
}



int main( void )
{
    std::srand( 87134623 );
    auto constexpr nTests = 100;
    auto constexpr nRepeatTests = 20; /* with different kernel configs */
    auto constexpr maxValueInTests = 128;

    int iDevice = 0;
    CUDA_ERROR( cudaGetDevice( &iDevice ) );
    cudaDeviceProp props;
    CUDA_ERROR( cudaGetDeviceProperties( &props, iDevice ) );
    auto const nMaxValuesPerTest = props.maxThreadsPerBlock; // 32

    MirroredVector< unsigned int > nToReduce( nTests );
    MirroredVector< unsigned int > dataToReduce( nMaxValuesPerTest * nTests );
    MirroredVector< unsigned int > resultsGpu ( nTests );
    MirroredVector< unsigned int > resultsHost( nTests );
    std::generate( nToReduce.host, nToReduce.host + nToReduce.nElements, [
                   =](){ return std::rand() % nMaxValuesPerTest; } );
    std::generate( dataToReduce.host, dataToReduce.host + dataToReduce.nElements, [
                   =](){ return std::rand() % maxValueInTests; } );
    dataToReduce.push();

    std::cout << "Test | GPU sum | Host sum\n";
    for ( auto iRepetition = 0u; iRepetition < nRepeatTests; ++iRepetition )
    {
        for ( auto iTest = 0u; iTest < nTests; ++iTest )
        {
            auto const nThreads = nToReduce.host[ iTest ];
            kernelTestSum<<< 1, nThreads >>>(
                nToReduce.host[ iTest ],
                dataToReduce.gpu + iTest * nMaxValuesPerTest,
                resultsGpu.gpu + iTest
            );
            CUDA_ERROR( cudaStreamSynchronize(0) );
            CUDA_ERROR( cudaPeekAtLastError() );
        }
        /* do summing on CPU */
        std::fill( resultsHost.host, resultsHost.host + resultsHost.nElements, 0 );
        for ( auto iTest = 0u; iTest < nTests; ++iTest )
        for ( auto i = 0u; i < nToReduce.host[ iTest ]; ++i )
            resultsHost.host[ iTest ] += dataToReduce.host[ iTest * nMaxValuesPerTest + i ];
        resultsGpu.pop();
        /* compare sums and print debug output */
        for ( auto iTest = 0u; iTest < nTests; ++iTest )
        {
            if ( resultsGpu.host[ iTest ] != resultsHost.host[ iTest ] )
            {
                std::cout
                << std::setw(5) << iTest << " "
                << std::setw(8) << resultsGpu .host[ iTest ] << "  "
                << std::setw(8) << resultsHost.host[ iTest ] << " => "
                << ( resultsGpu.host[ iTest ] == resultsHost.host[ iTest ] ? "OK" : "FAIL" )
                << "\n";
            }
            assert( resultsGpu.host[ iTest ] == resultsHost.host[ iTest ] );
        }
    }
}
