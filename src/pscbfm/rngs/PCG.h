#pragma once

/**
 * Please check the copyright conditions before you use this class.
 * pcg-random.org
 */

#include <stdint.h>

#ifdef __CUDACC__
#   define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#   define CUDA_CALLABLE_MEMBER
#endif


namespace Rngs {


class PCG
{
public:
    class State
    {
    public:
        uint64_t inc;
        uint64_t state;
        inline State( void ) : inc(0), state(0) {}
        inline State( uint64_t seed, uint64_t stream )
         : inc( stream*2+1 ), state(0)
        {
            PCG::pcg32_random( *this );
            state += seed;
            PCG::pcg32_random( *this );
            //Improve quality of first random numbers
            PCG::pcg32_random( *this );
        }
        inline ~State( void ){}
    };

    using GlobalState = PCG::State;

private:
    friend class State;
    CUDA_CALLABLE_MEMBER inline static uint32_t pcg32_random( PCG::State & rng )
    {
        const uint64_t old = rng.state;
        // Advance internal state
        rng.state = ((uint64_t) rng.state) * 0x5851F42D4C957F2DULL;
        rng.state += (rng.inc | 1);
        const uint32_t xorshifted = ((old >> 18u) ^ old) >> 27u;
        const uint32_t rot = old >> 59u;
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }

    State * my_state;

public:

    CUDA_CALLABLE_MEMBER inline  PCG( void ){}
    CUDA_CALLABLE_MEMBER inline ~PCG( void ){}

    CUDA_CALLABLE_MEMBER static constexpr bool needsSeed       ( void ){ return false; }
    CUDA_CALLABLE_MEMBER static constexpr bool needsSubsequence( void ){ return false; }
    CUDA_CALLABLE_MEMBER static constexpr bool needsIteration  ( void ){ return false; }
    CUDA_CALLABLE_MEMBER static constexpr bool needsGlobalState( void ){ return true ; }

    CUDA_CALLABLE_MEMBER inline void setGlobalState( PCG::GlobalState * ptr ){ my_state = ptr; }
    CUDA_CALLABLE_MEMBER inline void setIteration  ( uint64_t const ){}
    CUDA_CALLABLE_MEMBER inline void setSeed       ( uint64_t const ){}
    CUDA_CALLABLE_MEMBER inline void setSubsequence( uint64_t const ){}

    CUDA_CALLABLE_MEMBER inline uint32_t rng32(void){ return pcg32_random(*my_state); }
};


}
