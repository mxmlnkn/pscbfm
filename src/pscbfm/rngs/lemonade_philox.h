#pragma once

#include <stdint.h>
//#include <curand.h>
#include <curand_kernel.h>


#ifdef __CUDACC__
#   define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#   define CUDA_CALLABLE_MEMBER
#endif


namespace Rngs {


class lemonade_philox
{
private:
    uint64_t mSeed;
    uint64_t mIteration;
    uint64_t mSubseq;

    curandStatePhilox4_32_10_t c_state;
    //curandState c_state;
    bool initialized;

public:
    //Int instead of void to suppress compiler warnings
    using GlobalState = int;


    CUDA_CALLABLE_MEMBER inline  lemonade_philox( void ) : initialized( false ){}
    CUDA_CALLABLE_MEMBER inline ~lemonade_philox( void ){};

    CUDA_CALLABLE_MEMBER static constexpr bool needsSeed       ( void ){ return true ; }
    CUDA_CALLABLE_MEMBER static constexpr bool needsSubsequence( void ){ return true ; }
    CUDA_CALLABLE_MEMBER static constexpr bool needsIteration  ( void ){ return true ; }
    CUDA_CALLABLE_MEMBER static constexpr bool needsGlobalState( void ){ return false; }

    CUDA_CALLABLE_MEMBER inline void setSeed       ( uint64_t const rSeed      ){ mSeed      = rSeed     ; }
    CUDA_CALLABLE_MEMBER inline void setSubsequence( uint64_t const rSubseq    ){ mSubseq    = rSubseq   ; }
    CUDA_CALLABLE_MEMBER inline void setIteration  ( uint64_t const rIteration ){ mIteration = rIteration; }
    CUDA_CALLABLE_MEMBER inline void setGlobalState( void const * ){}

    //CUDA_CALLABLE_MEMBER uint32_t rng32(void)
    __device__ inline uint32_t rng32( void )
    {
        if ( ! initialized  )
        {
            curand_init( mSeed, mSubseq, mIteration, &c_state );
            initialized = true;
        }
        return curand( &c_state );
    }
};


}
