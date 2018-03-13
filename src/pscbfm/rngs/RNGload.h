#pragma once

#include <stdint.h>


#ifdef __CUDACC__
#   define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#   define CUDA_CALLABLE_MEMBER
#endif



/* This is a class to load pregenerated RNGs */
class RNGload
{
public:
    using GlobalState = uint32_t;

    CUDA_CALLABLE_MEMBER  RNGload( void ){}
    CUDA_CALLABLE_MEMBER ~RNGload( void ){}

    CUDA_CALLABLE_MEMBER static constexpr bool needsSeed       ( void ){ return false; }
    CUDA_CALLABLE_MEMBER static constexpr bool needsSubsequence( void ){ return false; }
    CUDA_CALLABLE_MEMBER static constexpr bool needsIteration  ( void ){ return false; }
    CUDA_CALLABLE_MEMBER static constexpr bool needsGlobalState( void ){ return true ; }

    CUDA_CALLABLE_MEMBER void setSeed       ( uint64_t const ){}
    CUDA_CALLABLE_MEMBER void setSubsequence( uint64_t const ){}
    CUDA_CALLABLE_MEMBER void setIteration  ( uint64_t const ){}
    CUDA_CALLABLE_MEMBER void setGlobalState( GlobalState const * ptr ){ mPtr = ptr; }

    CUDA_CALLABLE_MEMBER uint32_t rng32( void ){ return *mPtr; }

private:
    GlobalState const * mPtr;
};
