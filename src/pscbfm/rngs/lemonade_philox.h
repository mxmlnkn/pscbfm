#pragma once

#include <stdint.h>
//#include <curand.h>
#include <curand_kernel.h>


#ifdef __CUDACC__
#   define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#   define CUDA_CALLABLE_MEMBER
#endif



//This is just a template class, never use it
class lemonade_philox{
private:
        uint64_t mSeed;
        uint64_t mIteration;
        uint64_t mSubseq;

        curandStatePhilox4_32_10_t c_state;
        //curandState c_state;
       bool initialized;

public:
    //Int instead of void to suppress compiler warnings
    typedef int global_state_type;


    CUDA_CALLABLE_MEMBER  lemonade_philox( void ) : initialized( false ){}
    CUDA_CALLABLE_MEMBER ~lemonade_philox( void );

    CUDA_CALLABLE_MEMBER static constexpr bool needsSeed       ( void ){ return true ; }
    CUDA_CALLABLE_MEMBER static constexpr bool needsSubsequence( void ){ return true ; }
    CUDA_CALLABLE_MEMBER static constexpr bool needsIteration  ( void ){ return true ; }
    CUDA_CALLABLE_MEMBER static constexpr bool needsGlobalState( void ){ return false; }

    CUDA_CALLABLE_MEMBER void setSeed       ( uint64_t const rSeed      ){ mSeed      = rSeed     ; }
    CUDA_CALLABLE_MEMBER void setSubsequence( uint64_t const rSubseq    ){ mSubseq    = rSubseq   ; }
    CUDA_CALLABLE_MEMBER void setIteration  ( uint64_t const rIteration ){ mIteration = rIteration; }
    CUDA_CALLABLE_MEMBER void setGlobalState( void const * ){}

    //CUDA_CALLABLE_MEMBER uint32_t rng32(void)
    __device__ uint32_t rng32( void )
    {
        if ( ! initialized  )
        {
          curand_init( mSeed, mSubseq, mIteration, &c_state );
          initialized = true;
        }
        return curand( &c_state );
    }
};
