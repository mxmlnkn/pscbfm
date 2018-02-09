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
    unsigned int       * const __restrict__ dpResults     ,
    unsigned int         const              iToTest
)
{
    if ( dpnToReduce == 0 )
        *dpResults = 0;

    __shared__ int smBuffer[32];
    if ( threadIdx.x < dpnToReduce )
    {
        int sum = 0;
        switch ( iToTest )
        {
            case 0: sum =  warpReduceSum( (int) dpDataToReduce[ threadIdx.x ] ); break;
            case 1: sum = blockReduceSum( (int) dpDataToReduce[ threadIdx.x ], smBuffer ); break;
            case 2: sum =  warpReduceSumPredicate( (bool) dpDataToReduce[ threadIdx.x ] ); break;
            case 3: sum = blockReduceSumPredicate( (bool) dpDataToReduce[ threadIdx.x ], smBuffer ); break;
        }
        if ( threadIdx.x == 0 )
            *dpResults = sum;
    }
}



int main( void )
{
    std::srand( 87134623 );
    auto constexpr nTests = 100;
    auto constexpr nRepeatTests = 2; /* with different kernel configs */
    auto constexpr maxValueInTests = 128;

    int iDevice = 0;
    CUDA_ERROR( cudaGetDevice( &iDevice ) );
    cudaDeviceProp props;
    CUDA_ERROR( cudaGetDeviceProperties( &props, iDevice ) );

    /**
     * iKernel == 0 checks warpReduceSum, iKernel == 1 checks blockReduceSum
     * 2,3 as above, but for predicate
     * 4,5,6,7 same as the four above, but cumsum -> need to copy-paste, because will get too complicated
     */
    for ( int iKernel = 0; iKernel < 4; ++iKernel )
    {
        auto const nMaxValuesPerTest = ( iKernel % 2 == 0 ? 32 /* warp reduces */ : props.maxThreadsPerBlock );

        MirroredVector< unsigned int > nToReduce( nTests );
        MirroredVector< unsigned int > dataToReduce( nMaxValuesPerTest * nTests );
        MirroredVector< unsigned int > resultsGpu ( nTests );
        MirroredVector< unsigned int > resultsHost( nTests );
        std::generate( nToReduce.host, nToReduce.host + nToReduce.nElements, [
                       =](){ return std::rand() % nMaxValuesPerTest; } );
        std::generate( dataToReduce.host, dataToReduce.host + dataToReduce.nElements, [
                       =](){ return std::rand() % maxValueInTests; } );
        dataToReduce.push();

        std::vector< std::string > sTestNames = { "warpReduceSum", "blockReduceSum", "warpReduceSumPredicate", "blockReduceSumPredicate" };
        std::cout << "========= " << sTestNames[ iKernel ] << " =========\n";
        std::cout << "Kernel | (nBlocks,nThreads) | Test | GPU sum | Host sum (only showing fails)\n";
        for ( auto iRepetition = 0u; iRepetition < nRepeatTests; ++iRepetition )
        {
            for ( auto iTest = 0u; iTest < nTests; ++iTest )
            {
                auto const nBlocks = 1;
                auto const nThreads = std::max( 1u, nToReduce.host[ iTest ] );
                kernelTestSum<<< nBlocks, nThreads >>>(
                    nToReduce.host[ iTest ],
                    dataToReduce.gpu + iTest * nMaxValuesPerTest,
                    resultsGpu.gpu + iTest,
                    iKernel
                );
                CUDA_ERROR( cudaStreamSynchronize(0) );
                CUDA_ERROR( cudaPeekAtLastError() );
            }

            /* do summing on CPU */
            std::fill( resultsHost.host, resultsHost.host + resultsHost.nElements, 0 );
            for ( auto iTest = 0u; iTest < nTests; ++iTest )
            for ( auto i = 0u; i < nToReduce.host[ iTest ]; ++i )
            {
                switch ( iKernel )
                {
                    case 0: case 1:
                        resultsHost.host[ iTest ] += dataToReduce.host[ iTest * nMaxValuesPerTest + i ];
                        break;
                    case 2: case 3:
                        resultsHost.host[ iTest ] += (bool) dataToReduce.host[ iTest * nMaxValuesPerTest + i ];
                        break;
                }
            }
            resultsGpu.pop();

            /* compare sums and print debug output */
            for ( auto iTest = 0u; iTest < nTests; ++iTest )
            {
                auto const nBlocks = 1;
                auto const nThreads = nToReduce.host[ iTest ];
                if ( resultsGpu.host[ iTest ] != resultsHost.host[ iTest ] )
                {
                    std::cout
                    << std::setw(6) << iKernel << " | ("
                    << std::setw(10) << nBlocks << "," << std::setw(5) << nThreads << ") | "
                    << std::setw(4) << iTest << " | "
                    << std::setw(8) << resultsGpu .host[ iTest ] << "  "
                    << std::setw(8) << resultsHost.host[ iTest ] << " => "
                    << ( resultsGpu.host[ iTest ] == resultsHost.host[ iTest ] ? "OK" : "FAIL" )
                    << "\n";
                }
                assert( resultsGpu.host[ iTest ] == resultsHost.host[ iTest ] );
            }
        }
    }
}
