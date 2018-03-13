#ifndef RNG_TEMPLATE_H
#define RNG_TEMPLATE_H

#include <cstdint>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

//This is just a template class, never use it
class templateRNG{
public:
    //Int instead of void to suppress compiler warnings
    typedef int global_state_type;

    CUDA_CALLABLE_MEMBER templateRNG(void)
        {}

    CUDA_CALLABLE_MEMBER ~templateRNG(void)
        {}

    CUDA_CALLABLE_MEMBER static constexpr bool needs_global_state(void)
        {return false;}

    CUDA_CALLABLE_MEMBER void set_global_state(const global_state_type*ptr)
        {}

    CUDA_CALLABLE_MEMBER static constexpr bool needs_iteration(void)
        {return false;}

    CUDA_CALLABLE_MEMBER void set_iteration(const uint64_t iteration)
        {}

    CUDA_CALLABLE_MEMBER static constexpr bool needs_seed(void)
        {return false;}

    CUDA_CALLABLE_MEMBER void set_seed(const uint64_t seed)
        {}

    CUDA_CALLABLE_MEMBER static constexpr bool needs_subsequence(void)
        {return false;}

    CUDA_CALLABLE_MEMBER void set_subsequence(const uint64_t subseq)
        {}

    CUDA_CALLABLE_MEMBER uint32_t rng32(void)
        {return 0;}

    };

#endif//RNG_TEMPLATE_H
