/*
 * UpdaterGPUScBFM_AB_Type.cpp
 *
 *  Created on: 27.07.2017
 *      Author: Ron Dockhorn
 */

#include "UpdaterGPUScBFM_AB_Type.h"


//#define USE_THRUST_FILL
#define USE_BIT_PACKING_TMP_LATTICE
//#define USE_BIT_PACKING_LATTICE
//#define AUTO_CONFIGURE_BEST_SETTINGS_FOR_PSCBFM_ALGORITHM


#include <algorithm>                        // fill, sort
#include <chrono>                           // std::chrono::high_resolution_clock
#include <cstdio>                           // printf
#include <cstdlib>                          // exit
#include <cstring>                          // memset
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <stdint.h>
#include <sstream>

#include <cuda_profiler_api.h>              // cudaProfilerStop
#ifdef USE_THRUST_FILL
#   include <thrust/system/cuda/execution_policy.h>
#   include <thrust/fill.h>
#endif

#include "Fundamental/BitsCompileTime.hpp"

#include "cudacommon.hpp"
#include "SelectiveLogger.hpp"
#include "graphColoring.tpp"

#define DEBUG_UPDATERGPUSCBFM_AB_TYPE 100
#if defined( USE_BIT_PACKING_TMP_LATTICE ) || defined( USE_BIT_PACKING_LATTICE )
#   define USE_BIT_PACKING
#endif

/* 512=8^3 for a range of bonds per direction of [-4,3] */
__device__ __constant__ bool dpForbiddenBonds[512]; //false-allowed; true-forbidden

/**
 * These will be initialized to:
 *   DXTable_d = { -1,1,0,0,0,0 }
 *   DYTable_d = { 0,0,-1,1,0,0 }
 *   DZTable_d = { 0,0,0,0,-1,1 }
 * I.e. a table of three random directional 3D vectors \vec{dr} = (dx,dy,dz)
 */
__device__ __constant__ uint32_t DXTable_d[6]; //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z
__device__ __constant__ uint32_t DYTable_d[6]; //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z
__device__ __constant__ uint32_t DZTable_d[6]; //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z
/**
 * If intCUDA is different from uint32_t, then this second table prevents
 * expensive type conversions, but both tables are still needed, the
 * uint32_t version, because the calculation of the linear index will result
 * in uint32_t anyway and the intCUDA version for solely updating the
 * position information
 */
__device__ __constant__ intCUDA DXTableIntCUDA_d[6];
__device__ __constant__ intCUDA DYTableIntCUDA_d[6];
__device__ __constant__ intCUDA DZTableIntCUDA_d[6];

/* will this really bring performance improvement? At least constant cache
 * might be as fast as register access when all threads in a warp access the
 * the same constant */
__device__ __constant__ uint32_t dcBoxXM1   ;  // mLattice size in X-1
__device__ __constant__ uint32_t dcBoxYM1   ;  // mLattice size in Y-1
__device__ __constant__ uint32_t dcBoxZM1   ;  // mLattice size in Z-1
__device__ __constant__ uint32_t dcBoxXLog2 ;  // mLattice shift in X
__device__ __constant__ uint32_t dcBoxXYLog2;  // mLattice shift in X*Y


/* Since CUDA 5.5 (~2014) there do exist texture objects which are much
 * easier and can actually be used as kernel arguments!
 * @see https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-kepler-texture-objects-improve-performance-and-flexibility/
 * "What is not commonly known is that each outstanding texture reference that
 *  is bound when a kernel is launched incurs added launch latency—up to 0.5 μs
 *  per texture reference. This launch overhead persists even if the outstanding
 *  bound textures are not even referenced by the kernel. Again, using texture
 *  objects instead of texture references completely removes this overhead."
 * => they only exist for kepler -.- ...
 */

__device__ uint32_t hash( uint32_t a )
{
    /* https://web.archive.org/web/20120626084524/http://www.concentric.net:80/~ttwang/tech/inthash.htm
     * Note that before this 2007-03 version there were no magic numbers.
     * This hash function doesn't seem to be published.
     * He writes himself that this shouldn't really be used for PRNGs ???
     * @todo E.g. check random distribution of randomly drawn directions are
     *       they rouhgly even?
     * The 'hash' or at least an older version of it can even be inverted !!!
     * http://c42f.github.io/2015/09/21/inverting-32-bit-wang-hash.html
     * Somehow this also gets attibuted to Robert Jenkins?
     * https://gist.github.com/badboy/6267743
     * -> http://www.burtleburtle.net/bob/hash/doobs.html
     *    http://burtleburtle.net/bob/hash/integer.html
     */
    a = ( a + 0x7ed55d16 ) + ( a << 12 );
    a = ( a ^ 0xc761c23c ) ^ ( a >> 19 );
    a = ( a + 0x165667b1 ) + ( a << 5  );
    a = ( a + 0xd3a2646c ) ^ ( a << 9  );
    a = ( a + 0xfd7046c5 ) + ( a << 3  );
    a = ( a ^ 0xb55a4f09 ) ^ ( a >> 16 );
    return a;
}


/**
 * Morton / Z-curve ordering is a mapping N^n->N: (x,y,..)->i
 * When drawing a line by following i sequentially it looks like a
 * fractal Z-Curve:
 * @verbatim
 * y\x |      |   0  |   1  |   2  |   3
 * ----+------+------+------+------+------
 * dec | bin  | 0b00 | 0b01 | 0b10 | 0b11
 * ----+------+------+------+------+------
 *  0  | 0b00 | 0000 | 0001 | 0100 | 0101
 *     |      |   0  |   1  |   4  |   5
 *  1  | 0b01 | 0010 | 0011 | 0110 | 0111
 *     |      |   2  |   3  |   6  |   7
 *  2  | 0b10 | 1000 | 1001 | 1100 | 1101
 *     |      |   8  |   9  |  12  |  13
 *  3  | 0b11 | 1010 | 1011 | 1110 | 1111
 *     |      |  10  |  11  |  14  |  15
 * @endverbatim
 * As can be see from this:
 *  - The (x,y)->i follows a simple scheme in the bit representation,
 *    i.e. the bit representations of x and y get interleaved, for z
 *    and higher coordinates it would be similar
 *  - The grid must have a size of power of two in each dimension.
 *    Padding a grid would be problematic, as the padded unused
 *    cells are intermingled inbetween the used memory locations
 */

/* for a in-depth description see comments in Fundamental/BitsCompileTime.hpp
 * where it was copied from */
template< typename T, unsigned char nSpacing, unsigned char nStepsNeeded, unsigned char iStep >
struct DiluteBitsCrumble { __device__ __host__ inline static T apply( T const & xLastStep )
{
    auto x = DiluteBitsCrumble<T,nSpacing,nStepsNeeded,iStep-1>::apply( xLastStep );
    auto constexpr iStep2Pow = 1llu << ( (nStepsNeeded-1) - iStep );
    auto constexpr mask = BitPatterns::RectangularWave< T, iStep2Pow, iStep2Pow * nSpacing >::value;
    x = ( x | ( x << ( iStep2Pow * nSpacing ) ) ) & mask;
    return x;
} };

template< typename T, unsigned char nSpacing, unsigned char nStepsNeeded >
struct DiluteBitsCrumble<T,nSpacing,nStepsNeeded,0> { __device__ __host__ inline static T apply( T const & x )
{
    auto constexpr nBitsAllowed = 1 + ( sizeof(T) * CHAR_BIT - 1 ) / ( nSpacing + 1 );
    return x & BitPatterns::Ones< T, nBitsAllowed >::value;
} };

template< typename T, unsigned char nSpacing >
__device__ __host__ inline T diluteBits( T const & rx )
{
    static_assert( nSpacing > 0, "" );
    auto constexpr nBitsAvailable = sizeof(T) * CHAR_BIT;
    static_assert( nBitsAvailable > 0, "" );
    auto constexpr nBitsAllowed = CompileTimeFunctions::ceilDiv( nBitsAvailable, nSpacing + 1 );
    auto constexpr nStepsNeeded = 1 + CompileTimeFunctions::CeilLog< 2, nBitsAllowed >::value;
    return DiluteBitsCrumble< T, nSpacing, nStepsNeeded, ( nStepsNeeded > 0 ? nStepsNeeded-1 : 0 ) >::apply( rx );
}

/**
 * Legacy function which ironically might be more readable than my version
 * which derives and thereby documents in-code where the magic constants
 * derive from :(
 * Might be needed to compare performance to the template version.
 *  => is slower by 1%
 * Why is it using ^ instead of | ??? !!!
 */
/*
__device__ uint32_t part1by2_d( uint32_t n )
{
    n&= 0x000003ff;
    n = (n ^ (n << 16)) & 0xff0000ff; // 0b 0000 0000 1111 1111
    n = (n ^ (n <<  8)) & 0x0300f00f; // 0b 1111 0000 0000 1111
    n = (n ^ (n <<  4)) & 0x030c30c3; // 0b 0011 0000 1100 0011
    n = (n ^ (n <<  2)) & 0x09249249; // 0b 1001 0010 0100 1001
    return n;
}
*/

namespace {

template< typename T >
__device__ __host__ bool isPowerOfTwo( T const & x )
{
    //return ! ( x == T(0) ) && ! ( x & ( x - T(1) ) );
    return __popc( x ) <= 1;
}

}

#define USE_ZCURVE_FOR_LATTICE
uint32_t UpdaterGPUScBFM_AB_Type::linearizeBoxVectorIndex
(
    uint32_t const & ix,
    uint32_t const & iy,
    uint32_t const & iz
)
{
    #if defined ( USE_ZCURVE_FOR_LATTICE )
        return   diluteBits< uint32_t, 2 >( ix & mBoxXM1 )        |
               ( diluteBits< uint32_t, 2 >( iy & mBoxYM1 ) << 1 ) |
               ( diluteBits< uint32_t, 2 >( iz & mBoxZM1 ) << 2 );
    #elif defined( NOMAGIC )
        return ( ix % mBoxX ) +
               ( iy % mBoxY ) * mBoxX +
               ( iz % mBoxZ ) * mBoxX * mBoxY;
    #else
        #if DEBUG_UPDATERGPUSCBFM_AB_TYPE > 10
            assert( isPowerOfTwo( mBoxXM1 + 1 ) );
            assert( isPowerOfTwo( mBoxYM1 + 1 ) );
            assert( isPowerOfTwo( mBoxZM1 + 1 ) );
        #endif
        return   ( ix & mBoxXM1 ) +
               ( ( iy & mBoxYM1 ) << mBoxXLog2  ) +
               ( ( iz & mBoxZM1 ) << mBoxXYLog2 );
    #endif
}

__device__ inline uint32_t linearizeBoxVectorIndex
(
    uint32_t const & ix,
    uint32_t const & iy,
    uint32_t const & iz
)
{
    #if defined ( USE_ZCURVE_FOR_LATTICE )
        return   diluteBits< uint32_t, 2 >( ix & dcBoxXM1 )        |
               ( diluteBits< uint32_t, 2 >( iy & dcBoxYM1 ) << 1 ) |
               ( diluteBits< uint32_t, 2 >( iz & dcBoxZM1 ) << 2 );
    #else
        #if DEBUG_UPDATERGPUSCBFM_AB_TYPE > 10
            assert( isPowerOfTwo( dcBoxXM1 + 1 ) );
            assert( isPowerOfTwo( dcBoxYM1 + 1 ) );
            assert( isPowerOfTwo( dcBoxZM1 + 1 ) );
        #endif
        return   ( ix & dcBoxXM1 ) +
               ( ( iy & dcBoxYM1 ) << dcBoxXLog2  ) +
               ( ( iz & dcBoxZM1 ) << dcBoxXYLog2 );
    #endif
}

#define USE_BIT_PACKING
#ifdef USE_BIT_PACKING
    template< typename T > __device__ __host__ inline
    T bitPackedGet( T const * const & p, uint32_t const & i )
    {
        /**
         * >> 3, because 3 bits = 2^3=8 numbers are used for sub-byte indexing,
         * i.e. we divide the index i by 8 which is equal to the space we save
         * by bitpacking.
         * & 7, because 7 = 0b111, i.e. we are only interested in the last 3
         * bits specifying which subbyte element we want
         */
        return ( p[ i >> 3 ] >> ( i & T(7) ) ) & T(1);
    }

    template< typename T > __device__ inline
    T bitPackedTextureGet( cudaTextureObject_t const & p, uint32_t const & i )
    {
        return ( tex1Dfetch<T>( p, i >> 3 ) >> ( i & T(7) ) ) & T(1);
    }

    /**
     * Because the smalles atomic is for int (4x uint8_t) we need to
     * cast the array to that and then do a bitpacking for the whole 32 bits
     * instead of 8 bits
     * I.e. we need to address 32 subbits, i.e. >>3 becomes >>5
     * and &7 becomes &31 = 0b11111 = 0x1F
     * __host__ __device__ function with differing code
     * @see https://codeyarns.com/2011/03/14/cuda-common-function-for-both-host-and-device-code/
     */
    template< typename T > __device__ __host__ inline
    void bitPackedSet( T * const __restrict__ p, uint32_t const & i )
    {
        static_assert( sizeof(int) == 4, "" );
        #ifdef __CUDA_ARCH__
            atomicOr ( (int*) p + ( i >> 5 ),    T(1) << ( i & T( 0x1F ) )   );
        #else
            p[ i >> 3 ] |= T(1) << ( i & T(7) );
        #endif
    }

    template< typename T > __device__ __host__ inline
    void bitPackedUnset( T * const __restrict__ p, uint32_t const & i )
    {
        #ifdef __CUDA_ARCH__
            atomicAnd( (int*) p + ( i >> 5 ), ~( T(1) << ( i & T( 0x1F ) ) ) );
        #else
            p[ i >> 3 ] &= ~( T(1) << ( i & T(7) ) );
        #endif
    }
#else
    template< typename T > __device__ __host__ inline
    T bitPackedGet( T const * const & p, uint32_t const & i ){ return p[i]; }
    template< typename T > __device__ inline
    T bitPackedTextureGet( cudaTextureObject_t const & p, uint32_t const & i ) {
        return tex1Dfetch<T>(p,i); }
    template< typename T > __device__ __host__ inline
    void bitPackedSet  ( T * const __restrict__ p, uint32_t const & i ){ p[i] = 1; }
    template< typename T > __device__ __host__ inline
    void bitPackedUnset( T * const __restrict__ p, uint32_t const & i ){ p[i] = 0; }
#endif


/**
 * Checks the 3x3 grid one in front of the new position in the direction of the
 * move given by axis.
 *
 * @verbatim
 *           ____________
 *         .'  .'  .'  .'|
 *        +---+---+---+  +     y
 *        | 6 | 7 | 8 |.'|     ^ z
 *        +---+---+---+  +     |/
 *        | 3/| 4/| 5 |.'|     +--> x
 *        +-/-+-/-+---+  +
 *   0 -> |+---+1/| 2 |.'  ^          ^
 *        /|/-/|/-+---+   /          / axis direction +z (axis = 0b101)
 *       / +-/-+         /  2 (*dz) /                              ++|
 *      +---+ /         /                                         /  +/-
 *      |/X |/         L                                        xyz
 *      +---+  <- X ... current position of the monomer
 * @endverbatim
 *
 * @param[in] axis +-x, +-y, +-z in that order from 0 to 5, or put in another
 *                 equivalent way: the lowest bit specifies +(1) or -(0) and the
 *                 Bit 2 and 1 specify the axis: 0b00=x, 0b01=y, 0b10=z
 * @return Returns true if any of that is occupied, i.e. if there
 *         would be a problem with the excluded volume condition.
 */
__device__ inline bool checkFront
(
    cudaTextureObject_t const & texLattice,
    uint32_t            const & x0        ,
    uint32_t            const & y0        ,
    uint32_t            const & z0        ,
    intCUDA             const & axis
)
{
#if 0
    bool isOccupied = false;
    #define TMP_FETCH( x,y,z ) \
        tex1Dfetch< uint8_t >( texLattice, linearizeBoxVectorIndex(x,y,z) )
    auto const shift  = intCUDA(4) * ( axis & intCUDA(1) ) - intCUDA(2);
    auto const iMove = axis >> intCUDA(1);
    /* reduce branching by parameterizing the access axis, but that
     * makes the memory accesses more random again ???
     * for i0=0, i1=1, axis=z (same as in function doxygen ascii art)
     *    4 3 2
     *    5 0 1
     *    6 7 8
     */
    intCUDA r[3] = { x0, y0, z0 };
    r[ iMove ] += shift; isOccupied = TMP_FETCH( r[0], r[1], r[2] ); /* 0 */
    intCUDA i0 = iMove+1 >= 3 ? iMove+1-3 : iMove+1;
    intCUDA i1 = iMove+2 >= 3 ? iMove+2-3 : iMove+2;
    r[ i0 ]++; isOccupied |= TMP_FETCH( r[0], r[1], r[2] ); /* 1 */
    r[ i1 ]++; isOccupied |= TMP_FETCH( r[0], r[1], r[2] ); /* 2 */
    r[ i0 ]--; isOccupied |= TMP_FETCH( r[0], r[1], r[2] ); /* 3 */
    r[ i0 ]--; isOccupied |= TMP_FETCH( r[0], r[1], r[2] ); /* 4 */
    r[ i1 ]--; isOccupied |= TMP_FETCH( r[0], r[1], r[2] ); /* 5 */
    r[ i1 ]--; isOccupied |= TMP_FETCH( r[0], r[1], r[2] ); /* 6 */
    r[ i0 ]++; isOccupied |= TMP_FETCH( r[0], r[1], r[2] ); /* 7 */
    r[ i0 ]++; isOccupied |= TMP_FETCH( r[0], r[1], r[2] ); /* 8 */
    #undef TMP_FETCH
#elif 0 // defined( NOMAGIC )
    bool isOccupied = false;
    intCUDA const shift = 4*(axis & 1)-2;
    switch ( axis >> 1 )
    {
        #define TMP_FETCH( x,y,z ) \
            tex1Dfetch< uint8_t >( texLattice, linearizeBoxVectorIndex(x,y,z) )
        case 0: //-+x
        {
            uint32_t const x1 = x0 + shift;
            isOccupied = TMP_FETCH( x1, y0 - 1, z0     ) |
                         TMP_FETCH( x1, y0    , z0     ) |
                         TMP_FETCH( x1, y0 + 1, z0     ) |
                         TMP_FETCH( x1, y0 - 1, z0 - 1 ) |
                         TMP_FETCH( x1, y0    , z0 - 1 ) |
                         TMP_FETCH( x1, y0 + 1, z0 - 1 ) |
                         TMP_FETCH( x1, y0 - 1, z0 + 1 ) |
                         TMP_FETCH( x1, y0    , z0 + 1 ) |
                         TMP_FETCH( x1, y0 + 1, z0 + 1 );
            break;
        }
        case 1: //-+y
        {
            uint32_t const y1 = y0 + shift;
            isOccupied = TMP_FETCH( x0 - 1, y1, z0 - 1 ) |
                         TMP_FETCH( x0    , y1, z0 - 1 ) |
                         TMP_FETCH( x0 + 1, y1, z0 - 1 ) |
                         TMP_FETCH( x0 - 1, y1, z0     ) |
                         TMP_FETCH( x0    , y1, z0     ) |
                         TMP_FETCH( x0 + 1, y1, z0     ) |
                         TMP_FETCH( x0 - 1, y1, z0 + 1 ) |
                         TMP_FETCH( x0    , y1, z0 + 1 ) |
                         TMP_FETCH( x0 + 1, y1, z0 + 1 );
            break;
        }
        case 2: //-+z
        {
            /**
             * @verbatim
             *   +---+---+---+  y
             *   | 6 | 7 | 8 |  ^ z
             *   +---+---+---+  |/
             *   | 3 | 4 | 5 |  +--> x
             *   +---+---+---+
             *   | 0 | 1 | 2 |
             *   +---+---+---+
             * @endverbatim
             */
            uint32_t const z1 = z0 + shift;
            isOccupied = TMP_FETCH( x0 - 1, y0 - 1, z1 ) | /* 0 */
                         TMP_FETCH( x0    , y0 - 1, z1 ) | /* 1 */
                         TMP_FETCH( x0 + 1, y0 - 1, z1 ) | /* 2 */
                         TMP_FETCH( x0 - 1, y0    , z1 ) | /* 3 */
                         TMP_FETCH( x0    , y0    , z1 ) | /* 4 */
                         TMP_FETCH( x0 + 1, y0    , z1 ) | /* 5 */
                         TMP_FETCH( x0 - 1, y0 + 1, z1 ) | /* 6 */
                         TMP_FETCH( x0    , y0 + 1, z1 ) | /* 7 */
                         TMP_FETCH( x0 + 1, y0 + 1, z1 );  /* 8 */
            break;
        }
        #undef TMP_FETCH
    }
#else
    #if defined( USE_ZCURVE_FOR_LATTICE )
        auto const x0Abs  = diluteBits< uint32_t, 2 >( ( x0               ) & dcBoxXM1 );
        auto const x0PDX  = diluteBits< uint32_t, 2 >( ( x0 + uint32_t(1) ) & dcBoxXM1 );
        auto const x0MDX  = diluteBits< uint32_t, 2 >( ( x0 - uint32_t(1) ) & dcBoxXM1 );
        auto const y0Abs  = diluteBits< uint32_t, 2 >( ( y0               ) & dcBoxYM1 ) << 1;
        auto const y0PDY  = diluteBits< uint32_t, 2 >( ( y0 + uint32_t(1) ) & dcBoxYM1 ) << 1;
        auto const y0MDY  = diluteBits< uint32_t, 2 >( ( y0 - uint32_t(1) ) & dcBoxYM1 ) << 1;
        auto const z0Abs  = diluteBits< uint32_t, 2 >( ( z0               ) & dcBoxZM1 ) << 2;
        auto const z0PDZ  = diluteBits< uint32_t, 2 >( ( z0 + uint32_t(1) ) & dcBoxZM1 ) << 2;
        auto const z0MDZ  = diluteBits< uint32_t, 2 >( ( z0 - uint32_t(1) ) & dcBoxZM1 ) << 2;
    #else
        auto const x0Abs  =   ( x0               ) & dcBoxXM1;
        auto const x0PDX  =   ( x0 + uint32_t(1) ) & dcBoxXM1;
        auto const x0MDX  =   ( x0 - uint32_t(1) ) & dcBoxXM1;
        auto const y0Abs  = ( ( y0               ) & dcBoxYM1 ) << dcBoxXLog2;
        auto const y0PDY  = ( ( y0 + uint32_t(1) ) & dcBoxYM1 ) << dcBoxXLog2;
        auto const y0MDY  = ( ( y0 - uint32_t(1) ) & dcBoxYM1 ) << dcBoxXLog2;
        auto const z0Abs  = ( ( z0               ) & dcBoxZM1 ) << dcBoxXYLog2;
        auto const z0PDZ  = ( ( z0 + uint32_t(1) ) & dcBoxZM1 ) << dcBoxXYLog2;
        auto const z0MDZ  = ( ( z0 - uint32_t(1) ) & dcBoxZM1 ) << dcBoxXYLog2;
    #endif

    auto const dx = DXTable_d[ axis ];   // 2*axis-1
    auto const dy = DYTable_d[ axis ];   // 2*(axis&1)-1
    auto const dz = DZTable_d[ axis ];   // 2*(axis&1)-1

    uint32_t is[9];

    #if defined( USE_ZCURVE_FOR_LATTICE )
        switch ( axis >> intCUDA(1) )
        {
            case 0: is[7] = ( x0 + decltype(dx)(2) * dx ) & dcBoxXM1; break;
            case 1: is[7] = ( y0 + decltype(dy)(2) * dy ) & dcBoxYM1; break;
            case 2: is[7] = ( z0 + decltype(dz)(2) * dz ) & dcBoxZM1; break;
        }
        is[7] = diluteBits< uint32_t, 2 >( is[7] ) << ( axis >> intCUDA(1) );
    #else
        switch ( axis >> intCUDA(1) )
        {
            case 0: is[7] =   ( x0 + decltype(dx)(2) * dx ) & dcBoxXM1; break;
            case 1: is[7] = ( ( y0 + decltype(dy)(2) * dy ) & dcBoxYM1 ) << dcBoxXLog2; break;
            case 2: is[7] = ( ( z0 + decltype(dz)(2) * dz ) & dcBoxZM1 ) << dcBoxXYLog2; break;
        }
    #endif
    switch ( axis >> intCUDA(1) )
    {
        case 0: //-+x
        {
            is[2]  = is[7] | z0Abs;
            is[5]  = is[7] | z0MDZ;
            is[8]  = is[7] | z0PDZ;
            is[0]  = is[2] | y0MDY;
            is[1]  = is[2] | y0Abs;
            is[2] |=         y0PDY;
            is[3]  = is[5] | y0MDY;
            is[4]  = is[5] | y0Abs;
            is[5] |=         y0PDY;
            is[6]  = is[8] | y0MDY;
            is[7]  = is[8] | y0Abs;
            is[8] |=         y0PDY;
            break;
        }
        case 1: //-+y
        {
            is[2]  = is[7] | z0MDZ;
            is[5]  = is[7] | z0Abs;
            is[8]  = is[7] | z0PDZ;
            is[0]  = is[2] | x0MDX;
            is[1]  = is[2] | x0Abs;
            is[2] |=         x0PDX;
            is[3]  = is[5] | x0MDX;
            is[4]  = is[5] | x0Abs;
            is[5] |=         x0PDX;
            is[6]  = is[8] | x0MDX;
            is[7]  = is[8] | x0Abs;
            is[8] |=         x0PDX;
            break;
        }
        case 2: //-+z
        {
            is[2]  = is[7] | y0MDY;
            is[5]  = is[7] | y0Abs;
            is[8]  = is[7] | y0PDY;
            is[0]  = is[2] | x0MDX;
            is[1]  = is[2] | x0Abs;
            is[2] |=         x0PDX;
            is[3]  = is[5] | x0MDX;
            is[4]  = is[5] | x0Abs;
            is[5] |=         x0PDX;
            is[6]  = is[8] | x0MDX;
            is[7]  = is[8] | x0Abs;
            is[8] |=         x0PDX;
            break;
        }
    }
    bool const isOccupied = tex1Dfetch< uint8_t >( texLattice, is[0] ) |
                            tex1Dfetch< uint8_t >( texLattice, is[1] ) |
                            tex1Dfetch< uint8_t >( texLattice, is[2] ) |
                            tex1Dfetch< uint8_t >( texLattice, is[3] ) |
                            tex1Dfetch< uint8_t >( texLattice, is[4] ) |
                            tex1Dfetch< uint8_t >( texLattice, is[5] ) |
                            tex1Dfetch< uint8_t >( texLattice, is[6] ) |
                            tex1Dfetch< uint8_t >( texLattice, is[7] ) |
                            tex1Dfetch< uint8_t >( texLattice, is[8] );
#endif
    return isOccupied;
}

#ifdef USE_BIT_PACKING
__device__ inline bool checkFrontBitPacked
(
    cudaTextureObject_t const & texLattice,
    uint32_t            const & x0        ,
    uint32_t            const & y0        ,
    uint32_t            const & z0        ,
    intCUDA             const & axis
)
{
    auto const x0Abs  = diluteBits< uint32_t, 2 >( ( x0               ) & dcBoxXM1 );
    auto const x0PDX  = diluteBits< uint32_t, 2 >( ( x0 + uint32_t(1) ) & dcBoxXM1 );
    auto const x0MDX  = diluteBits< uint32_t, 2 >( ( x0 - uint32_t(1) ) & dcBoxXM1 );
    auto const y0Abs  = diluteBits< uint32_t, 2 >( ( y0               ) & dcBoxYM1 ) << 1;
    auto const y0PDY  = diluteBits< uint32_t, 2 >( ( y0 + uint32_t(1) ) & dcBoxYM1 ) << 1;
    auto const y0MDY  = diluteBits< uint32_t, 2 >( ( y0 - uint32_t(1) ) & dcBoxYM1 ) << 1;
    auto const z0Abs  = diluteBits< uint32_t, 2 >( ( z0               ) & dcBoxZM1 ) << 2;
    auto const z0PDZ  = diluteBits< uint32_t, 2 >( ( z0 + uint32_t(1) ) & dcBoxZM1 ) << 2;
    auto const z0MDZ  = diluteBits< uint32_t, 2 >( ( z0 - uint32_t(1) ) & dcBoxZM1 ) << 2;

    auto const dx = DXTable_d[ axis ];   // 2*axis-1
    auto const dy = DYTable_d[ axis ];   // 2*(axis&1)-1
    auto const dz = DZTable_d[ axis ];   // 2*(axis&1)-1

    uint32_t is[9];
    switch ( axis >> intCUDA(1) )
    {
        case 0: is[7] = ( x0 + decltype(dx)(2) * dx ) & dcBoxXM1; break;
        case 1: is[7] = ( y0 + decltype(dy)(2) * dy ) & dcBoxYM1; break;
        case 2: is[7] = ( z0 + decltype(dz)(2) * dz ) & dcBoxZM1; break;
    }
    is[7] = diluteBits< uint32_t, 2 >( is[7] ) << ( axis >> intCUDA(1) );
    switch ( axis >> intCUDA(1) )
    {
        case 0: //-+x
            is[2]  = is[7] | z0Abs; is[5]  = is[7] | z0MDZ; is[8]  = is[7] | z0PDZ;
            is[0]  = is[2] | y0MDY; is[1]  = is[2] | y0Abs; is[2] |=         y0PDY;
            is[3]  = is[5] | y0MDY; is[4]  = is[5] | y0Abs; is[5] |=         y0PDY;
            is[6]  = is[8] | y0MDY; is[7]  = is[8] | y0Abs; is[8] |=         y0PDY;
            break;
        case 1: //-+y
            is[2]  = is[7] | z0MDZ; is[5]  = is[7] | z0Abs; is[8]  = is[7] | z0PDZ;
            is[0]  = is[2] | x0MDX; is[1]  = is[2] | x0Abs; is[2] |=         x0PDX;
            is[3]  = is[5] | x0MDX; is[4]  = is[5] | x0Abs; is[5] |=         x0PDX;
            is[6]  = is[8] | x0MDX; is[7]  = is[8] | x0Abs; is[8] |=         x0PDX;
            break;
        case 2: //-+z
            is[2]  = is[7] | y0MDY; is[5]  = is[7] | y0Abs; is[8]  = is[7] | y0PDY;
            is[0]  = is[2] | x0MDX; is[1]  = is[2] | x0Abs; is[2] |=         x0PDX;
            is[3]  = is[5] | x0MDX; is[4]  = is[5] | x0Abs; is[5] |=         x0PDX;
            is[6]  = is[8] | x0MDX; is[7]  = is[8] | x0Abs; is[8] |=         x0PDX;
            break;
    }
    return bitPackedTextureGet< uint8_t >( texLattice, is[0] ) |
           bitPackedTextureGet< uint8_t >( texLattice, is[1] ) |
           bitPackedTextureGet< uint8_t >( texLattice, is[2] ) |
           bitPackedTextureGet< uint8_t >( texLattice, is[3] ) |
           bitPackedTextureGet< uint8_t >( texLattice, is[4] ) |
           bitPackedTextureGet< uint8_t >( texLattice, is[5] ) |
           bitPackedTextureGet< uint8_t >( texLattice, is[6] ) |
           bitPackedTextureGet< uint8_t >( texLattice, is[7] ) |
           bitPackedTextureGet< uint8_t >( texLattice, is[8] );
}
#endif

__device__ __host__ inline uintCUDA linearizeBondVectorIndex
(
    intCUDA const x,
    intCUDA const y,
    intCUDA const z
)
{
    /* Just like for normal integers we clip the range to go more down than up
     * i.e. [-127 ,128] or in this case [-4,3]
     * +4 maps to the same location as -4 but is needed or else forbidden
     * bonds couldn't be detected. Larger bonds are not possible, because
     * monomers only move by 1 per step */
    //assert( -4 <= x && x <= 4 );
    //assert( -4 <= y && y <= 4 );
    //assert( -4 <= z && z <= 4 );
    return   ( x & intCUDA(7) /* 0b111 */ ) +
           ( ( y & intCUDA(7) /* 0b111 */ ) << intCUDA(3) ) +
           ( ( z & intCUDA(7) /* 0b111 */ ) << intCUDA(6) );
}

/**
 * Goes over all monomers of a species given specified by texSpeciesIndices
 * draws a random direction for them and checks whether that move is possible
 * with the box size and periodicity as well as the monomers at the target
 * location (excluded volume) and the new bond lengths to all neighbors.
 * If so, then the new position is set to 1 in dpLatticeTmp and encode the
 * possible movement direction in the property tag of the corresponding monomer
 * in dpPolymerSystem.
 * Note that the old position is not removed in order to correctly check for
 * excluded volume a second time.
 *
 * @param[in] rn a random number used as a kind of seed for the RNG
 * @param[in] nMonomers number of max. monomers to work on, this is for
 *            filtering out excessive threads and was prior a __constant__
 *            But it is only used one(!) time in the kernel so the caching
 *            of constant memory might not even be used.
 *            @see https://web.archive.org/web/20140612185804/http://www.pixel.io/blog/2013/5/9/kernel-arguments-vs-__constant__-variables.html
 *            -> Kernel arguments are even put into constant memory it seems:
 *            @see "Section E.2.5.2 Function Parameters" in the "CUDA 5.5 C Programming Guide"
 *            __global__ function parameters are passed to the device:
 *             - via shared memory and are limited to 256 bytes on devices of compute capability 1.x,
 *             - via constant memory and are limited to 4 KB on devices of compute capability 2.x and higher.
 *            __device__ and __global__ functions cannot have a variable number of arguments.
 * @param[in] iOffset Offste to curent species we are supposed to work on.
 *            We can't simply adjust dpPolymerSystem before calling the kernel,
 *            because we are accessing the neighbors, therefore need all the
 *            polymer data, especially for other species.
 *
 * Note: all of the three kernels do quite few work. They basically just fetch
 *       data, and check one condition and write out again. There isn't even
 *       a loop and most of the work seems to be boiler plate initialization
 *       code which could be cut if the kernels could be merged together.
 *       Why are there three kernels instead of just one
 *        -> for global synchronization
 */
using T_Flags = UpdaterGPUScBFM_AB_Type::T_Flags;
__global__ void kernelSimulationScBFMCheckSpecies
(
    intCUDA     const * const __restrict__ dpPolymerSystem         ,
    T_Flags           * const __restrict__ dpPolymerFlags          ,
    uint32_t            const              nPolymerSystemPitch     ,
    uint32_t            const              iOffset                 ,
    uint8_t           * const __restrict__ dpLatticeTmp            ,
    uint32_t    const * const __restrict__ dpNeighbors             ,
    uint32_t            const              rNeighborsPitchElements ,
    uint8_t     const * const __restrict__ dpNeighborsSizes        ,
    uint32_t            const              nMonomers               ,
    uint32_t            const              rSeed                   ,
    cudaTextureObject_t const              texLatticeRefOut
)
{
    uint32_t rn;
    int iGrid = 0;
    for ( auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x, ++iGrid )
    {
        CudaVec3< intCUDA >::value_type r0;
        r0.x = dpPolymerSystem[ iOffset + iMonomer ];
        r0.y = dpPolymerSystem[ iOffset + iMonomer + nPolymerSystemPitch ];
        r0.z = dpPolymerSystem[ iOffset + iMonomer + nPolymerSystemPitch + nPolymerSystemPitch ];
        /* upcast int3 to int4 in preparation to use PTX SIMD instructions */
        //int4 const r0 = { r0Raw.x, r0Raw.y, r0Raw.z, 0 }; // not faster nor slower
        //select random direction. Own implementation of an rng :S? But I think it at least# was initialized using the LeMonADE RNG ...
        if ( iGrid % 1 == 0 ) // 12 = floor( log(2^32) / log(6) )
            rn = hash( hash( iMonomer ) ^ rSeed );
        T_Flags const direction = rn % T_Flags(6); rn /= T_Flags(6);
        T_Flags properties = 0;

         /* select random direction. Do this with bitmasking instead of lookup ??? */
        /* int4 const dr = { DXTable_d[ direction ],
                          DYTable_d[ direction ],
                          DZTable_d[ direction ], 0 }; */
        uint3 const r1 = { r0.x + DXTable_d[ direction ],
                           r0.y + DYTable_d[ direction ],
                           r0.z + DZTable_d[ direction ] };

    #ifdef NONPERIODICITY
       /* check whether the new location of the particle would be inside the box
        * if the box is not periodic, if not, then don't move the particle */
        if ( uint32_t(0) <= r1.x && r1.x < dcBoxXM1 &&
             uint32_t(0) <= r1.y && r1.y < dcBoxYM1 &&
             uint32_t(0) <= r1.z && r1.z < dcBoxZM1    )
        {
    #endif
            /* check whether the new position would result in invalid bonds
             * between this monomer and its neighbors */
            auto const nNeighbors = dpNeighborsSizes[ iOffset + iMonomer ];
            bool forbiddenBond = false;
            for ( auto iNeighbor = decltype( nNeighbors )(0); iNeighbor < nNeighbors; ++iNeighbor )
            {
                auto const iGlobalNeighbor = dpNeighbors[ iNeighbor * rNeighborsPitchElements + iMonomer ];
                CudaVec3< intCUDA >::value_type data2;
                data2.x = dpPolymerSystem[ iGlobalNeighbor ];
                data2.y = dpPolymerSystem[ iGlobalNeighbor + nPolymerSystemPitch ];
                data2.z = dpPolymerSystem[ iGlobalNeighbor + nPolymerSystemPitch + nPolymerSystemPitch ];
                if ( dpForbiddenBonds[ linearizeBondVectorIndex( data2.x - r1.x, data2.y - r1.y, data2.z - r1.z ) ] )
                {
                    forbiddenBond = true;
                    break;
                }
            }
            if ( ! forbiddenBond && ! checkFront( texLatticeRefOut, r0.x, r0.y, r0.z, direction ) )
            {
                /* everything fits so perform move on temporary lattice */
                /* can I do this ??? dpPolymerSystem is the device pointer to the read-only
                 * texture used above. Won't this result in read-after-write race-conditions?
                 * Then again the written / changed bits are never used in the above code ... */
                properties = ( direction << T_Flags(2) ) + T_Flags(1) /* can-move-flag */;
            #ifdef USE_BIT_PACKING_TMP_LATTICE
                bitPackedSet( dpLatticeTmp, linearizeBoxVectorIndex( r1.x, r1.y, r1.z ) );
            #else
                dpLatticeTmp[ linearizeBoxVectorIndex( r1.x, r1.y, r1.z ) ] = 1;
            #endif
            }
    #ifdef NONPERIODICITY
        }
    #endif
        dpPolymerFlags[ iOffset + iMonomer ] = properties;
    }
}


__global__ void kernelCountFilteredCheck
(
    intCUDA          const * const __restrict__ dpPolymerSystem        ,
    T_Flags          const * const __restrict__ dpPolymerFlags         ,
    uint32_t                 const              nPolymerSystemPitch    ,
    uint32_t                 const              iOffset                ,
    uint8_t          const * const __restrict__ /* dpLatticeTmp */     ,
    uint32_t         const * const __restrict__ dpNeighbors            ,
    uint32_t                 const              rNeighborsPitchElements,
    uint8_t          const * const __restrict__ dpNeighborsSizes       ,
    uint32_t                 const              nMonomers              ,
    uint32_t                 const              rSeed                  ,
    cudaTextureObject_t      const              texLatticeRefOut       ,
    unsigned long long int * const __restrict__ dpFiltered
)
{
    uint32_t rn;
    int iGrid = 0;
    for ( auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x, ++iGrid )
    {
        CudaVec3< intCUDA >::value_type r0;
        r0.x = dpPolymerSystem[ iOffset + iMonomer ];
        r0.y = dpPolymerSystem[ iOffset + iMonomer + nPolymerSystemPitch ];
        r0.z = dpPolymerSystem[ iOffset + iMonomer + nPolymerSystemPitch + nPolymerSystemPitch ];
        if ( iGrid % 1 == 0 )
            rn = hash( hash( iMonomer ) ^ rSeed );
        T_Flags const direction = rn % T_Flags(6); rn /= T_Flags(6);

        uint3 const r1 = { r0.x + DXTable_d[ direction ],
                           r0.y + DYTable_d[ direction ],
                           r0.z + DZTable_d[ direction ] };

    #ifdef NONPERIODICITY
        if ( uint32_t(0) <= r1.x && r1.x < dcBoxXM1 &&
             uint32_t(0) <= r1.y && r1.y < dcBoxYM1 &&
             uint32_t(0) <= r1.z && r1.z < dcBoxZM1    )
        {
    #endif
            auto const nNeighbors = dpNeighborsSizes[ iOffset + iMonomer ];
            bool forbiddenBond = false;
            for ( auto iNeighbor = decltype( nNeighbors )(0); iNeighbor < nNeighbors; ++iNeighbor )
            {
                auto const iGlobalNeighbor = dpNeighbors[ iNeighbor * rNeighborsPitchElements + iMonomer ];
                CudaVec3< intCUDA >::value_type data2;
                data2.x = dpPolymerSystem[ iGlobalNeighbor ];
                data2.y = dpPolymerSystem[ iGlobalNeighbor + nPolymerSystemPitch ];
                data2.z = dpPolymerSystem[ iGlobalNeighbor + nPolymerSystemPitch + nPolymerSystemPitch ];
                if ( dpForbiddenBonds[ linearizeBondVectorIndex( data2.x - r1.x, data2.y - r1.y, data2.z - r1.z ) ] )
                {
                    atomicAdd( dpFiltered+1, 1 );
                    forbiddenBond = true;
                    break;
                }
            }

            if ( checkFront( texLatticeRefOut, r0.x, r0.y, r0.z, direction ) )
            {
                atomicAdd( dpFiltered+2, 1 );
                if ( ! forbiddenBond ) /* this is the more real relative use-case where invalid bonds are already filtered out */
                    atomicAdd( dpFiltered+3, 1 );
            }
    #ifdef NONPERIODICITY
        }
    #endif
    }
}


/**
 * Recheck whether the move is possible without collision, using the
 * temporarily parallel executed moves saved in texLatticeTmp. If so,
 * do the move in dpLattice. (Still not applied in dpPolymerSystem!)
 */
__global__ void kernelSimulationScBFMPerformSpecies
(
    intCUDA       const * const __restrict__ dpPolymerSystem    ,
    T_Flags             * const __restrict__ dpPolymerFlags     ,
    uint32_t              const              nPolymerSystemPitch,
    uint8_t             * const __restrict__ dpLattice          ,
    uint32_t              const              nMonomers          ,
    cudaTextureObject_t   const              texLatticeTmp
)
{
    for ( auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x )
    {
        auto const properties = dpPolymerFlags[ iMonomer ];
        if ( ( properties & T_Flags(1) ) == T_Flags(0) )    // impossible move
            continue;

        CudaVec3< intCUDA >::value_type r0;
        r0.x = dpPolymerSystem[ iMonomer ];
        r0.y = dpPolymerSystem[ iMonomer + nPolymerSystemPitch ];
        r0.z = dpPolymerSystem[ iMonomer + nPolymerSystemPitch + nPolymerSystemPitch ];
        //uint3 const r0 = { r0Raw.x, r0Raw.y, r0Raw.z }; // slower
        auto const direction = ( properties >> T_Flags(2) ) & T_Flags(7); // 7=0b111
    #ifdef USE_BIT_PACKING_TMP_LATTICE
        if ( checkFrontBitPacked( texLatticeTmp, r0.x, r0.y, r0.z, direction ) )
    #else
        if ( checkFront( texLatticeTmp, r0.x, r0.y, r0.z, direction ) )
    #endif
            continue;

        /* If possible, perform move now on normal lattice */
        dpPolymerFlags[ iMonomer ] = properties | T_Flags(2); // indicating allowed move
        dpLattice[ linearizeBoxVectorIndex( r0.x, r0.y, r0.z ) ] = 0;
        dpLattice[ linearizeBoxVectorIndex( r0.x + DXTable_d[ direction ],
                                            r0.y + DYTable_d[ direction ],
                                            r0.z + DZTable_d[ direction ] ) ] = 1;
        /* We can't clean the temporary lattice in here, because it still is being
         * used for checks. For cleaning we need only the new positions.
         * But we can't use the applied positions, because we also need to clean
         * those particles which couldn't move in this second kernel but where
         * still set in the lattice by the first kernel! */
    }
}

__global__ void kernelSimulationScBFMPerformSpeciesAndApply
(
    intCUDA             * const __restrict__ dpPolymerSystem    ,
    T_Flags             * const __restrict__ dpPolymerFlags     ,
    uint32_t              const              nPolymerSystemPitch,
    uint8_t             * const __restrict__ dpLattice          ,
    uint32_t              const              nMonomers          ,
    cudaTextureObject_t   const              texLatticeTmp
)
{
    for ( auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x )
    {
        auto const properties = dpPolymerFlags[ iMonomer ];
        if ( ( properties & T_Flags(1) ) == T_Flags(0) )    // impossible move
            continue;

        CudaVec3< intCUDA >::value_type r0;
        r0.x = dpPolymerSystem[ iMonomer ];
        r0.y = dpPolymerSystem[ iMonomer + nPolymerSystemPitch ];
        r0.z = dpPolymerSystem[ iMonomer + nPolymerSystemPitch + nPolymerSystemPitch ];
        auto const direction = ( properties >> T_Flags(2) ) & T_Flags(7); // 7=0b111
    #ifdef USE_BIT_PACKING_TMP_LATTICE
        if ( checkFrontBitPacked( texLatticeTmp, r0.x, r0.y, r0.z, direction ) )
    #else
        if ( checkFront( texLatticeTmp, r0.x, r0.y, r0.z, direction ) )
    #endif
            continue;

        CudaVec4< intCUDA >::value_type const r1 = {
            intCUDA( r0.x + DXTableIntCUDA_d[ direction ] ),
            intCUDA( r0.y + DYTableIntCUDA_d[ direction ] ),
            intCUDA( r0.z + DZTableIntCUDA_d[ direction ] ), 0
        };
        /* If possible, perform move now on normal lattice */
        dpLattice[ linearizeBoxVectorIndex( r0.x, r0.y, r0.z ) ] = 0;
        dpLattice[ linearizeBoxVectorIndex( r1.x, r1.y, r1.z ) ] = 1;
        dpPolymerSystem[ iMonomer ]                                             = r1.x;
        dpPolymerSystem[ iMonomer + nPolymerSystemPitch ]                       = r1.y;
        dpPolymerSystem[ iMonomer + nPolymerSystemPitch + nPolymerSystemPitch ] = r1.z;
    }
}

__global__ void kernelCountFilteredPerform
(
    intCUDA          const * const __restrict__ dpPolymerSystem     ,
    T_Flags          const * const __restrict__ dpPolymerFlags      ,
    uint32_t                 const              nPolymerSystemPitch ,
    uint8_t          const * const __restrict__ /* dpLattice */     ,
    uint32_t                 const              nMonomers           ,
    cudaTextureObject_t      const              texLatticeTmp       ,
    unsigned long long int * const __restrict__ dpFiltered
)
{
    for ( auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x )
    {
        auto const properties = dpPolymerFlags[ iMonomer ];
        if ( ( properties & T_Flags(1) ) == T_Flags(0) )    // impossible move
            continue;

        CudaVec3< intCUDA >::value_type r0;
        r0.x = dpPolymerSystem[ iMonomer ];
        r0.y = dpPolymerSystem[ iMonomer + nPolymerSystemPitch ];
        r0.z = dpPolymerSystem[ iMonomer + nPolymerSystemPitch + nPolymerSystemPitch ];
        auto const direction = ( properties >> T_Flags(2) ) & T_Flags(7); // 7=0b111
        if ( checkFront( texLatticeTmp, r0.x, r0.y, r0.z, direction ) )
            atomicAdd( dpFiltered+4, size_t(1) );
    }
}

/**
 * Apply move to dpPolymerSystem and clean the temporary lattice of moves
 * which seemed like they would work, but did clash with another parallel
 * move, unfortunately.
 * @todo it might be better to just use a cudaMemset to clean the lattice,
 *       that way there wouldn't be any memory dependencies and calculations
 *       needed, even though we would have to clean everything, instead of
 *       just those set. But that doesn't matter, because most of the threads
 *       are idling anyway ...
 *       This kind of kernel might give some speedup after stream compaction
 *       has been implemented though.
 *    -> print out how many percent of cells need to be cleaned .. I need
 *       many more statistics anyway for evaluating performance benefits a bit
 *       better!
 */
__global__ void kernelSimulationScBFMZeroArraySpecies
(
    intCUDA             * const __restrict__ dpPolymerSystem    ,
    T_Flags       const * const __restrict__ dpPolymerFlags     ,
    uint32_t              const              nPolymerSystemPitch,
    uint8_t             * const __restrict__ dpLatticeTmp       ,
    uint32_t              const              nMonomers
)
{
    for ( auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x )
    {
        auto const properties = dpPolymerFlags[ iMonomer ];
        if ( ( properties & T_Flags(3) ) == T_Flags(0) )    // impossible move
            continue;

        CudaVec3< intCUDA >::value_type r0;
        r0.x = dpPolymerSystem[ iMonomer ];
        r0.y = dpPolymerSystem[ iMonomer + nPolymerSystemPitch ];
        r0.z = dpPolymerSystem[ iMonomer + nPolymerSystemPitch + nPolymerSystemPitch ];
        auto const direction = ( properties >> T_Flags(2) ) & T_Flags(7); // 7=0b111

        r0.x += DXTableIntCUDA_d[ direction ];
        r0.y += DYTableIntCUDA_d[ direction ];
        r0.z += DZTableIntCUDA_d[ direction ];
    #ifdef USE_BIT_PACKING_TMP_LATTICE
        //bitPackedUnset( dpLatticeTmp, linearizeBoxVectorIndex( r0.x, r0.y, r0.z ) );
        dpLatticeTmp[ linearizeBoxVectorIndex( r0.x, r0.y, r0.z ) >> 3 ] = 0;
    #else
        dpLatticeTmp[ linearizeBoxVectorIndex( r0.x, r0.y, r0.z ) ] = 0;
    #endif
        if ( ( properties & T_Flags(3) ) == T_Flags(3) )  // 3=0b11
        {
            dpPolymerSystem[ iMonomer ]                                             = r0.x;
            dpPolymerSystem[ iMonomer + nPolymerSystemPitch ]                       = r0.y;
            dpPolymerSystem[ iMonomer + nPolymerSystemPitch + nPolymerSystemPitch ] = r0.z;
        }
    }
}


UpdaterGPUScBFM_AB_Type::UpdaterGPUScBFM_AB_Type()
 : mStream              ( 0 ),
   nAllMonomers         ( 0 ),
   mLattice             ( NULL ),
   mLatticeOut          ( NULL ),
   mLatticeTmp          ( NULL ),
   mPolymerSystemSorted ( NULL ),
   mPolymerFlags        ( NULL ),
   mNeighborsSorted     ( NULL ),
   mNeighborsSortedSizes( NULL ),
   mNeighborsSortedInfo ( nBytesAlignment ),
   mAttributeSystem     ( NULL ),
   mBoxX                ( 0 ),
   mBoxY                ( 0 ),
   mBoxZ                ( 0 ),
   mBoxXM1              ( 0 ),
   mBoxYM1              ( 0 ),
   mBoxZM1              ( 0 ),
   mBoxXLog2            ( 0 ),
   mBoxXYLog2           ( 0 )
{
    /**
     * Log control.
     * Note that "Check" controls not the output, but the actualy checks
     * If a checks needs to always be done, then do that check and declare
     * the output as "Info" log level
     */
    mLog.file( __FILENAME__ );
    mLog.  activate( "Benchmark" );
    mLog.deactivate( "Check"     );
    mLog.  activate( "Error"     );
    mLog.  activate( "Info"      );
    mLog.deactivate( "Stats"     );
    mLog.deactivate( "Warning"   );
}

/**
 * Deletes everything which could and is allocated
 */
void UpdaterGPUScBFM_AB_Type::destruct()
{
    if ( mLattice         != NULL ){ delete[] mLattice        ; mLattice         = NULL; }  // setLatticeSize
    if ( mLatticeOut      != NULL ){ delete   mLatticeOut     ; mLatticeOut      = NULL; }  // initialize
    if ( mLatticeTmp      != NULL ){ delete   mLatticeTmp     ; mLatticeTmp      = NULL; }  // initialize
    if ( mPolymerSystemSorted != NULL ){ delete mPolymerSystemSorted; mPolymerSystemSorted = NULL; }  // initialize
    if ( mPolymerFlags    != NULL ){ delete   mPolymerFlags   ; mPolymerFlags    = NULL; }  // initialize
    if ( mNeighborsSorted != NULL ){ delete   mNeighborsSorted; mNeighborsSorted = NULL; }  // initialize
    if ( mNeighborsSortedSizes != NULL ){ delete   mNeighborsSortedSizes; mNeighborsSortedSizes = NULL; }  // initialize
    if ( mAttributeSystem != NULL ){ delete[] mAttributeSystem; mAttributeSystem = NULL; }  // setNrOfAllMonomers
}

UpdaterGPUScBFM_AB_Type::~UpdaterGPUScBFM_AB_Type()
{
    this->destruct();
}

void UpdaterGPUScBFM_AB_Type::setGpu( int iGpuToUse )
{
    int nGpus;
    getCudaDeviceProperties( NULL, &nGpus, true /* print GPU information */ );
    if ( ! ( 0 <= iGpuToUse && iGpuToUse < nGpus ) )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setGpu] "
            << "GPU with ID " << iGpuToUse << " not present. "
            << "Only " << nGpus << " GPUs are available.\n";
        mLog( "Error" ) << msg.str();
        throw std::invalid_argument( msg.str() );
    }
    CUDA_ERROR( cudaSetDevice( iGpuToUse ) );
    miGpuToUse = iGpuToUse;
}


void UpdaterGPUScBFM_AB_Type::initialize( void )
{
    if ( mLog( "Stats" ).isActive() )
    {
        // this is called in parallel it seems, therefore need to buffer it
        std::stringstream msg; msg
        << "[" << __FILENAME__ << "::initialize] The "
        << "(" << mBoxX << "," << mBoxY << "," << mBoxZ << ")"
        << " lattice is populated by " << nAllMonomers
        << " resulting in a filling rate of "
        << nAllMonomers / double( mBoxX * mBoxY * mBoxZ ) << "\n";
        mLog( "Stats" ) << msg.str();
    }

    if ( mLatticeOut != NULL || mLatticeTmp != NULL )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::initialize] "
            << "Initialize was already called and may not be called again "
            << "until cleanup was called!";
        mLog( "Error" ) << msg.str();
        throw std::runtime_error( msg.str() );
    }

    /* create the BondTable and copy it to constant memory */
    bool * tmpForbiddenBonds = (bool*) malloc( sizeof( bool ) * 512 );
    unsigned nAllowedBonds = 0;
    for ( int i = 0; i < 512; ++i )
        if ( ! ( tmpForbiddenBonds[i] = mForbiddenBonds[i] ) )
            ++nAllowedBonds;
    /* Why does it matter? Shouldn't it work with arbitrary bond sets ??? */
    if ( nAllowedBonds != 108 )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::initialize] "
            << "Wrong bond-set! Expected 108 allowed bonds, but got " << nAllowedBonds << "\n";
        mLog( "Error" ) << msg.str();
        throw std::runtime_error( msg.str() );
    }
    CUDA_ERROR( cudaMemcpyToSymbol( dpForbiddenBonds, tmpForbiddenBonds, sizeof(bool)*512 ) );
    free( tmpForbiddenBonds );

    /* create a table mapping the random int to directions whereto move the
     * monomers. We can use negative numbers, because (0u-1u)+1u still is 0u */
    uint32_t tmp_DXTable[6] = { 0u-1u,1,  0,0,  0,0 };
    uint32_t tmp_DYTable[6] = {  0,0, 0u-1u,1,  0,0 };
    uint32_t tmp_DZTable[6] = {  0,0,  0,0, 0u-1u,1 };
    CUDA_ERROR( cudaMemcpyToSymbol( DXTable_d, tmp_DXTable, sizeof( tmp_DXTable ) ) );
    CUDA_ERROR( cudaMemcpyToSymbol( DYTable_d, tmp_DYTable, sizeof( tmp_DXTable ) ) );
    CUDA_ERROR( cudaMemcpyToSymbol( DZTable_d, tmp_DZTable, sizeof( tmp_DXTable ) ) );
    intCUDA tmp_DXTableIntCUDA[6] = { -1,1,  0,0,  0,0 };
    intCUDA tmp_DYTableIntCUDA[6] = {  0,0, -1,1,  0,0 };
    intCUDA tmp_DZTableIntCUDA[6] = {  0,0,  0,0, -1,1 };
    CUDA_ERROR( cudaMemcpyToSymbol( DXTableIntCUDA_d, tmp_DXTableIntCUDA, sizeof( tmp_DZTableIntCUDA ) ) );
    CUDA_ERROR( cudaMemcpyToSymbol( DYTableIntCUDA_d, tmp_DYTableIntCUDA, sizeof( tmp_DZTableIntCUDA ) ) );
    CUDA_ERROR( cudaMemcpyToSymbol( DZTableIntCUDA_d, tmp_DZTableIntCUDA, sizeof( tmp_DZTableIntCUDA ) ) );

    /*************************** start of grouping ***************************/

   mLog( "Info" ) << "Coloring graph ...\n";
    bool const bUniformColors = true; // setting this to true should yield more performance as the kernels are uniformly utilized
    mGroupIds = graphColoring< MonomerEdges const *, uint32_t, uint8_t >(
        &mNeighbors[0], mNeighbors.size(), bUniformColors,
        []( MonomerEdges const * const & x, uint32_t const & i ){ return x[i].size; },
        []( MonomerEdges const * const & x, uint32_t const & i, size_t const & j ){ return x[i].neighborIds[j]; }
    );

    /* check automatic coloring with that given in BFM-file */
    if ( mLog.isActive( "Check" ) )
    {
        mLog( "Info" ) << "Checking difference between automatic and given coloring ... ";
        size_t nDifferent = 0;
        for ( size_t iMonomer = 0u; iMonomer < std::max( (uint32_t) 20, this->nAllMonomers ); ++iMonomer )
        {
            if ( mGroupIds.at( iMonomer )+1 != mAttributeSystem[ iMonomer ] )
            {
                 mLog( "Info" ) << "Color of " << iMonomer << " is automatically " << mGroupIds.at( iMonomer )+1 << " vs. " << mAttributeSystem[ iMonomer ] << "\n";
                ++nDifferent;
            }
        }
        if ( nDifferent > 0 )
        {
            std::stringstream msg;
            msg << "Automatic coloring failed to produce same result as the given one!";
            mLog( "Error" ) << msg.str();
            throw std::runtime_error( msg.str() );
        }
        mLog( "Info" ) << "OK\n";
    }

    /* count monomers per species before allocating per species arrays and
     * sorting the monomers into them */
    mLog( "Info" ) << "Attributes of first monomers: ";
    mnElementsInGroup.resize(0);
    for ( size_t i = 0u; i < mGroupIds.size(); ++i )
    {
        if ( i < 40 )
            mLog( "Info" ) << char( 'A' + (char) mGroupIds[i] );
        if ( mGroupIds[i] >= mnElementsInGroup.size() )
            mnElementsInGroup.resize( mGroupIds[i]+1, 0 );
        ++mnElementsInGroup[ mGroupIds[i] ];
    }
    mLog( "Info" ) << "\n";
    if ( mLog.isActive( "Stats" ) )
    {
        mLog( "Stats" ) << "Found " << mnElementsInGroup.size() << " groups with the frequencies: ";
        for ( size_t i = 0u; i < mnElementsInGroup.size(); ++i )
        {
            mLog( "Stats" ) << char( 'A' + (char) i ) << ": " << mnElementsInGroup[i] << "x (" << (float) mnElementsInGroup[i] / nAllMonomers * 100.f << "%), ";
        }
        mLog( "Stats" ) << "\n";
    }

    /**
     * Generate new array which contains all sorted monomers aligned
     * @verbatim
     * ABABABABABA
     * A A A A A A
     *  B B B B B
     * AAAAAA  BBBBB
     *        ^ alignment
     * @endverbatim
     * in the worst case we are only one element ( 4*intCUDA ) over the
     * alignment with each group and need to fill up to nBytesAlignment for
     * all of them */
    /* virtual number of monomers which includes the additional alignment padding */
    auto const nMonomersPadded = nAllMonomers + ( nElementsAlignment - 1u ) * mnElementsInGroup.size();

    /* calculate offsets to each aligned subgroup vector */
    iSubGroupOffset.resize( mnElementsInGroup.size() );
    iSubGroupOffset.at(0) = 0;
    for ( size_t i = 1u; i < mnElementsInGroup.size(); ++i )
    {
        iSubGroupOffset[i] = iSubGroupOffset[i-1] +
        ceilDiv( mnElementsInGroup[i-1], nElementsAlignment ) * nElementsAlignment;
        assert( iSubGroupOffset[i] - iSubGroupOffset[i-1] >= mnElementsInGroup[i-1] );
    }

    /* virtually sort groups into new array and save index mappings */
    iToiNew.resize( nAllMonomers   , UINT32_MAX );
    iNewToi.resize( nMonomersPadded, UINT32_MAX );
    std::vector< size_t > iSubGroup = iSubGroupOffset;   /* stores the next free index for each subgroup */
    for ( size_t i = 0u; i < nAllMonomers; ++i )
    {
        iToiNew[i] = iSubGroup[ mGroupIds[i] ]++;
        iNewToi[ iToiNew[i] ] = i;
    }

    if ( mLog.isActive( "Info" ) )
    {
        mLog( "Info" ) << "iSubGroupOffset = { ";
        for ( auto const & x : iSubGroupOffset )
            mLog( "Info" ) << x << ", ";
        mLog( "Info" ) << "}\n";

        mLog( "Info" ) << "iSubGroup = { ";
        for ( auto const & x : iSubGroup )
            mLog( "Info" ) << x << ", ";
        mLog( "Info" ) << "}\n";

        mLog( "Info" ) << "mnElementsInGroup = { ";
        for ( auto const & x : mnElementsInGroup )
            mLog( "Info" ) << x << ", ";
        mLog( "Info" ) << "}\n";
    }

    /* adjust neighbor IDs to new sorted PolymerSystem and also sort that array.
     * Bonds are not supposed to change, therefore we don't need to push and
     * pop them each time we do something on the GPU! */
    assert( mNeighborsSorted == NULL );
    assert( mNeighborsSortedInfo.getRequiredBytes() == 0 );
    for ( size_t i = 0u; i < mnElementsInGroup.size(); ++i )
        mNeighborsSortedInfo.newMatrix( MAX_CONNECTIVITY, mnElementsInGroup[i] );
    mNeighborsSorted = new MirroredVector< uint32_t >( mNeighborsSortedInfo.getRequiredElements(), mStream );
    std::memset( mNeighborsSorted->host, 0, mNeighborsSorted->nBytes );
    mNeighborsSortedSizes = new MirroredVector< uint8_t >( nMonomersPadded, mStream );
    std::memset( mNeighborsSortedSizes->host, 0, mNeighborsSortedSizes->nBytes );

    if ( mLog.isActive( "Info" ) )
    {
        mLog( "Info" )
        << "Allocated mNeighborsSorted with "
        << mNeighborsSorted->nElements << " elements in "
        << mNeighborsSorted->nBytes << "B ("
        << mNeighborsSortedInfo.getRequiredElements() << ","
        << mNeighborsSortedInfo.getRequiredBytes() << "B)\n";

        mLog( "Info" ) << "mNeighborsSortedInfo:\n";
        for ( size_t iSpecies = 0u; iSpecies < mnElementsInGroup.size(); ++iSpecies )
        {
            mLog( "Info" )
            << "== matrix/species " << iSpecies << " ==\n"
            << "offset:" << mNeighborsSortedInfo.getMatrixOffsetElements( iSpecies ) << " elements = "
                         << mNeighborsSortedInfo.getMatrixOffsetBytes   ( iSpecies ) << "B\n"
            //<< "rows  :" << mNeighborsSortedInfo.getOffsetElements() << " elements = "
            //             << mNeighborsSortedInfo.getOffsetBytes() << "B\n"
            //<< "cols  :" << mNeighborsSortedInfo.getOffsetElements() << " elements = "
            //             << mNeighborsSortedInfo.getOffsetBytes() << "B\n"
            << "pitch :" << mNeighborsSortedInfo.getMatrixPitchElements( iSpecies ) << " elements = "
                         << mNeighborsSortedInfo.getMatrixPitchBytes   ( iSpecies ) << "B\n";
        }
        mLog( "Info" ) << "[UpdaterGPUScBFM_AB_Type::runSimulationOnGPU] map neighborIds to sorted array ... ";
    }

    {
        size_t iSpecies = 0u;
        /* iterate over sorted instead of unsorted array so that calculating
         * the current species we are working on is easier */
        for ( size_t i = 0u; i < iNewToi.size(); ++i )
        {
            /* check if we are already working on a new species */
            if ( iSpecies+1 < iSubGroupOffset.size() &&
                 i >= iSubGroupOffset[ iSpecies+1 ] )
            {
                mLog( "Info" ) << "Currently at index " << i << "/" << iNewToi.size() << " and crossed offset of species " << iSpecies+1 << " at " << iSubGroupOffset[ iSpecies+1 ] << " therefore incrementing iSpecies\n";
                ++iSpecies;
            }
            /* skip over padded indices */
            if ( iNewToi[i] >= nAllMonomers )
                continue;
            /* actually to the sorting / copying and conversion */
            mNeighborsSortedSizes->host[i] = mNeighbors[ iNewToi[i] ].size;
            auto const pitch = mNeighborsSortedInfo.getMatrixPitchElements( iSpecies );
            for ( size_t j = 0u; j < mNeighbors[  iNewToi[i] ].size; ++j )
            {
                if ( i < 5 || std::abs( (long long int) i - iSubGroupOffset[ iSubGroupOffset.size()-1 ] ) < 5 )
                {
                    mLog( "Info" ) << "Currently at index " << i << ": Writing into mNeighborsSorted->host[ " << mNeighborsSortedInfo.getMatrixOffsetElements( iSpecies ) << " + " << j << " * " << pitch << " + " << i << "-" << iSubGroupOffset[ iSpecies ] << "] the value of old neighbor located at iToiNew[ mNeighbors[ iNewToi[i]=" << iNewToi[i] << " ] = iToiNew[ " << mNeighbors[ iNewToi[i] ].neighborIds[j] << " ] = " << iToiNew[ mNeighbors[ iNewToi[i] ].neighborIds[j] ] << " \n";
                }
                mNeighborsSorted->host[ mNeighborsSortedInfo.getMatrixOffsetElements( iSpecies ) + j * pitch + ( i - iSubGroupOffset[ iSpecies ] ) ] = iToiNew[ mNeighbors[ iNewToi[i] ].neighborIds[j] ];
                //mNeighborsSorted->host[ iToiNew[i] ].neighborIds[j] = iToiNew[ mNeighbors[i].neighborIds[j] ];
            }
        }
    }
    mNeighborsSorted->pushAsync();
    mNeighborsSortedSizes->pushAsync();
    mLog( "Info" ) << "Done\n";

    /* some checks for correctness of new adjusted neighbor global IDs */
    if ( mLog.isActive( "Check" ) )
    {
        /* note that this also checks "unitialized entries" those should be
         * initialized to 0 to reduce problems. This is done by the memset. */
        /*for ( size_t i = 0u; i < mNeighborsSorted->nElements; ++i )
        {
            if ( mNeighbors[i].size > MAX_CONNECTIVITY )
                throw std::runtime_error( "A monomer has more neighbors than allowed!" );
            for ( size_t j = 0u; j < mNeighbors[i].size; ++j )
            {
                auto const iSorted = mNeighborsSorted->host[i].neighborIds[j];
                if ( iSorted == UINT32_MAX )
                    throw std::runtime_error( "New index mapping not set!" );
                if ( iSorted >= nMonomersPadded )
                    throw std::runtime_error( "New index out of range!" );
            }
        }*/
        /* does a similar check for the unsorted error which is still used
         * to create the property tag */
        for ( uint32_t i = 0; i < nAllMonomers; ++i )
        {
            if ( mNeighbors[i].size > MAX_CONNECTIVITY )
            {
                std::stringstream msg;
                msg << "[" << __FILENAME__ << "::initialize] "
                    << "This implementation allows max. 7 neighbors per monomer, "
                    << "but monomer " << i << " has " << mNeighbors[i].size << "\n";
                mLog( "Error" ) << msg.str();
                throw std::invalid_argument( msg.str() );
            }
        }
    }

    assert( mPolymerFlags == NULL );
    mPolymerFlags = new MirroredVector< T_Flags >( nMonomersPadded, mStream );
    CUDA_ERROR( cudaMemset( mPolymerFlags->gpu, 0, mPolymerFlags->nBytes ) );

    assert( mPolymerSystemSortedInfo.getRequiredBytes() == 0 );
    /* still doing this manually, else the index mapping would become even more crazy ... */
    //for ( size_t i = 0u; i < mnElementsInGroup.size() /* nSpecies */; ++i )
    //    mPolymerSystemSortedInfo.newMatrix( 3 /* 3 rows for 3 coordinates in 3D */, mnElementsInGroup[i] );

    assert( mPolymerSystemSorted == NULL );
    mPolymerSystemSorted = new MirroredVector< intCUDA >( 3 * nMonomersPadded, mStream );
    #ifndef NDEBUG
        std::memset( mPolymerSystemSorted.host, 0, mPolymerSystemSorted.nBytes );
    #endif

    /* sort groups into new array and save index mappings */
    mLog( "Info" ) << "[UpdaterGPUScBFM_AB_Type::runSimulationOnGPU] sort mPolymerSystem -> mPolymerSystemSorted ... ";
    for ( size_t i = 0u; i < nAllMonomers; ++i )
    {
        if ( i < 20 )
            mLog( "Info" ) << "Write " << i << " to " << this->iToiNew[i] << "\n";
        mPolymerSystemSorted->host[ 0 * nMonomersPadded + iToiNew[i] ] = mPolymerSystem[4*i+0];
        mPolymerSystemSorted->host[ 1 * nMonomersPadded + iToiNew[i] ] = mPolymerSystem[4*i+1];
        mPolymerSystemSorted->host[ 2 * nMonomersPadded + iToiNew[i] ] = mPolymerSystem[4*i+2];
    }
    mPolymerSystemSorted->pushAsync();

    /************************** end of group sorting **************************/

    checkSystem();

    /* creating lattice */
    CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXM1   , &mBoxXM1   , sizeof( mBoxXM1    ) ) );
    CUDA_ERROR( cudaMemcpyToSymbol( dcBoxYM1   , &mBoxYM1   , sizeof( mBoxYM1    ) ) );
    CUDA_ERROR( cudaMemcpyToSymbol( dcBoxZM1   , &mBoxZM1   , sizeof( mBoxZM1    ) ) );
    CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXLog2 , &mBoxXLog2 , sizeof( mBoxXLog2  ) ) );
    CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXYLog2, &mBoxXYLog2, sizeof( mBoxXYLog2 ) ) );

    mLatticeOut = new MirroredTexture< uint8_t >( mBoxX * mBoxY * mBoxZ, mStream );
    mLatticeTmp = new MirroredTexture< uint8_t >( mBoxX * mBoxY * mBoxZ, mStream );
    CUDA_ERROR( cudaMemsetAsync( mLatticeTmp->gpu, 0, mLatticeTmp->nBytes, mStream ) );
    /* populate latticeOut with monomers from mPolymerSystem */
    std::memset( mLatticeOut->host, 0, mLatticeOut->nBytes );
    for ( uint32_t t = 0; t < nAllMonomers; ++t )
    {
        mLatticeOut->host[ linearizeBoxVectorIndex( mPolymerSystem[ 4*t+0 ],
                                                    mPolymerSystem[ 4*t+1 ],
                                                    mPolymerSystem[ 4*t+2 ] ) ] = 1;
    }
    mLatticeOut->pushAsync();

    mLog( "Info" )
        << "Filling Rate: " << nAllMonomers << " "
        << "(=" << nAllMonomers / 1024 << "*1024+" << nAllMonomers % 1024 << ") "
        << "particles in a (" << mBoxX << "," << mBoxY << "," << mBoxZ << ") box "
        << "=> " << 100. * nAllMonomers / ( mBoxX * mBoxY * mBoxZ ) << "%\n"
        << "Note: densest packing is: 25% -> in this case it might be more reasonable to actually iterate over the spaces where particles can move to, keeping track of them instead of iterating over the particles\n";

    CUDA_ERROR( cudaGetDevice( &miGpuToUse ) );
    CUDA_ERROR( cudaGetDeviceProperties( &mCudaProps, miGpuToUse ) );
}


void UpdaterGPUScBFM_AB_Type::copyBondSet
( int dx, int dy, int dz, bool bondForbidden )
{
    mForbiddenBonds[ linearizeBondVectorIndex(dx,dy,dz) ] = bondForbidden;
}

void UpdaterGPUScBFM_AB_Type::setNrOfAllMonomers( uint32_t const rnAllMonomers )
{
    if ( this->nAllMonomers != 0 || mAttributeSystem != NULL ||
         mPolymerSystemSorted != NULL || mNeighborsSorted != NULL )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setNrOfAllMonomers] "
            << "Number of Monomers already set to " << nAllMonomers << "!\n"
            << "Or some arrays were already allocated "
            << "(mAttributeSystem=" << (void*) mAttributeSystem
            << ", mPolymerSystemSorted" << (void*) mPolymerSystemSorted
            << ", mNeighborsSorted" << (void*) mNeighborsSorted << ")\n";
        throw std::runtime_error( msg.str() );
    }

    this->nAllMonomers = rnAllMonomers;
    mAttributeSystem = new int32_t[ nAllMonomers ];
    if ( mAttributeSystem == NULL )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setNrOfAllMonomers] mAttributeSystem is still NULL after call to 'new int32_t[ " << nAllMonomers << " ]!\n";
        mLog( "Error" ) << msg.str();
        throw std::runtime_error( msg.str() );
    }
    mPolymerSystem.resize( nAllMonomers*4 );
    mNeighbors    .resize( nAllMonomers   );
    std::memset( &mNeighbors[0], 0, mNeighbors.size() * sizeof( mNeighbors[0] ) );
}

void UpdaterGPUScBFM_AB_Type::setPeriodicity
(
    bool const isPeriodicX,
    bool const isPeriodicY,
    bool const isPeriodicZ
)
{
    /* Compare inputs to hardcoded values. No ability yet to adjust dynamically */
#ifdef NONPERIODICITY
    if ( isPeriodicX || isPeriodicY || isPeriodicZ )
#else
    if ( ! isPeriodicX || ! isPeriodicY || ! isPeriodicZ )
#endif
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setPeriodicity" << "] "
            << "Simulation is intended to use completely "
        #ifdef NONPERIODICITY
            << "non-"
        #endif
            << "periodic boundary conditions, but setPeriodicity was called with "
            << "(" << isPeriodicX << "," << isPeriodicY << "," << isPeriodicZ << ")\n";
        mLog( "Error" ) << msg.str();
        throw std::invalid_argument( msg.str() );
    }
}

void UpdaterGPUScBFM_AB_Type::setAttribute( uint32_t i, int32_t attribute )
{
    #if ! defined( NDEBUG ) && false
        std::cerr << "[setAttribute] mAttributeSystem = " << (void*) mAttributeSystem << "\n";
        if ( mAttributeSystem == NULL )
            throw std::runtime_error( "[UpdaterGPUScBFM_AB_Type.h::setAttribute] mAttributeSystem is NULL! Did you call setNrOfAllMonomers, yet?" );
        std::cerr << "set " << i << " to attribute " << attribute << "\n";
        if ( ! ( i < nAllMonomers ) )
            throw std::invalid_argument( "[UpdaterGPUScBFM_AB_Type.h::setAttribute] i out of range!" );
    #endif
    mAttributeSystem[i] = attribute;
}

void UpdaterGPUScBFM_AB_Type::setMonomerCoordinates
(
    uint32_t const i,
    int32_t  const x,
    int32_t  const y,
    int32_t  const z
)
{
#if DEBUG_UPDATERGPUSCBFM_AB_TYPE > 1
    /* can I apply periodic modularity here to allow the full range ??? */
    if ( ! inRange< decltype( mPolymerSystem[0] ) >(x) ||
         ! inRange< decltype( mPolymerSystem[0] ) >(y) ||
         ! inRange< decltype( mPolymerSystem[0] ) >(z)    )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setMonomerCoordinates" << "] "
            << "One or more of the given coordinates "
            << "(" << x << "," << y << "," << z << ") "
            << "is larger than the internal integer data type for "
            << "representing positions allow! (" << std::numeric_limits< intCUDA >::min()
            << " <= size <= " << std::numeric_limits< intCUDA >::max() << ")";
        throw std::invalid_argument( msg.str() );
    }
#endif
    mPolymerSystem.at( 4*i+0 ) = x;
    mPolymerSystem.at( 4*i+1 ) = y;
    mPolymerSystem.at( 4*i+2 ) = z;
}

int32_t UpdaterGPUScBFM_AB_Type::getMonomerPositionInX( uint32_t i ){ return mPolymerSystem[ 4*i+0 ]; }
int32_t UpdaterGPUScBFM_AB_Type::getMonomerPositionInY( uint32_t i ){ return mPolymerSystem[ 4*i+1 ]; }
int32_t UpdaterGPUScBFM_AB_Type::getMonomerPositionInZ( uint32_t i ){ return mPolymerSystem[ 4*i+2 ]; }

void UpdaterGPUScBFM_AB_Type::setConnectivity
(
    uint32_t const iMonomer1,
    uint32_t const iMonomer2
)
{
    /* @todo add check whether the bond already exists */
    /* Could also add the inversio, but the bonds are a non-directional graph */
    auto const iNew = mNeighbors[ iMonomer1 ].size++;
    if ( iNew > MAX_CONNECTIVITY-1 )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setConnectivity" << "] "
            << "The maximum amount of bonds per monomer (" << MAX_CONNECTIVITY
            << ") has been exceeded!\n";
        throw std::invalid_argument( msg.str() );
    }
    mNeighbors[ iMonomer1 ].neighborIds[ iNew ] = iMonomer2;
}

void UpdaterGPUScBFM_AB_Type::setLatticeSize
(
    uint32_t const boxX,
    uint32_t const boxY,
    uint32_t const boxZ
)
{
    if ( mBoxX == boxX && mBoxY == boxY && mBoxZ == boxZ )
        return;

    if ( ! ( inRange< intCUDA >( boxX ) &&
             inRange< intCUDA >( boxY ) &&
             inRange< intCUDA >( boxZ )    ) )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setLatticeSize" << "] "
            << "The box size (" << boxX << "," << boxY << "," << boxZ
            << ") is larger than the internal integer data type for "
            << "representing positions allow! (" << std::numeric_limits< intCUDA >::min()
            << " <= size <= " << std::numeric_limits< intCUDA >::max() << ")";
        throw std::invalid_argument( msg.str() );
    }

    mBoxX   = boxX;
    mBoxY   = boxY;
    mBoxZ   = boxZ;
    mBoxXM1 = boxX-1;
    mBoxYM1 = boxY-1;
    mBoxZM1 = boxZ-1;

    /* determine log2 for mBoxX and mBoxX * mBoxY to be used for bitshifting
     * the indice instead of multiplying ... WHY??? I don't think it is faster,
     * but much less readable */
    mBoxXLog2  = 0; uint32_t dummy = mBoxX; while ( dummy >>= 1 ) ++mBoxXLog2;
    mBoxXYLog2 = 0; dummy = mBoxX*mBoxY;    while ( dummy >>= 1 ) ++mBoxXYLog2;
    if ( mBoxX != ( 1u << mBoxXLog2 ) || mBoxX * boxY != ( 1u << mBoxXYLog2 ) )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setLatticeSize" << "] "
            << "Could not determine value for bit shift. "
            << "Check whether the box size is a power of 2! ( "
            << "boxX=" << mBoxX << " =? 2^" << mBoxXLog2 << " = " << ( 1 << mBoxXLog2 )
            << ", boxX*boY=" << mBoxX * mBoxY << " =? 2^" << mBoxXYLog2
            << " = " << ( 1 << mBoxXYLog2 ) << " )\n";
        throw std::runtime_error( msg.str() );
    }

    if ( mLattice != NULL )
        delete[] mLattice;
    mLattice = new uint8_t[ mBoxX * mBoxY * mBoxZ ];
    std::memset( (void *) mLattice, 0, mBoxX * mBoxY * mBoxZ * sizeof( *mLattice ) );
}

void UpdaterGPUScBFM_AB_Type::populateLattice()
{
    std::memset( mLattice, 0, mBoxX * mBoxY * mBoxZ * sizeof( *mLattice ) );
    for ( size_t i = 0; i < nAllMonomers; ++i )
    {
        auto const j = linearizeBoxVectorIndex( mPolymerSystem[ 4*i+0 ],
                                                mPolymerSystem[ 4*i+1 ],
                                                mPolymerSystem[ 4*i+2 ] );
        if ( j >= mBoxX * mBoxY * mBoxZ )
        {
            std::stringstream msg;
            msg
            << "[populateLattice] " << i << " -> ("
            << mPolymerSystem[ 4*i+0 ] << ","
            << mPolymerSystem[ 4*i+1 ] << ","
            << mPolymerSystem[ 4*i+2 ] << ") -> " << j << " is out of range "
            << "of (" << mBoxX << "," << mBoxY << "," << mBoxZ << ") = "
            << mBoxX * mBoxY * mBoxZ << "\n";
            throw std::runtime_error( msg.str() );
        }
        mLattice[ j ] = 1;
    }
}

/**
 * Checks for excluded volume condition and for correctness of all monomer bonds
 * Beware, it useses and thereby thrashes mLattice. Might be cleaner to declare
 * as const and malloc and free some temporary buffer, but the time ...
 * https://randomascii.wordpress.com/2014/12/10/hidden-costs-of-memory-allocation/
 * "In my tests, for sizes ranging from 8 MB to 32 MB, the cost for a new[]/delete[] pair averaged about 7.5 μs (microseconds), split into ~5.0 μs for the allocation and ~2.5 μs for the free."
 *  => ~40k cycles
 */
void UpdaterGPUScBFM_AB_Type::checkSystem()
{
    if ( ! mLog.isActive( "Check" ) )
        return;

    if ( mLattice == NULL )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::checkSystem" << "] "
            << "mLattice is not allocated. You need to call "
            << "setNrOfAllMonomers and initialize before calling checkSystem!\n";
        mLog( "Error" ) << msg.str();
        throw std::invalid_argument( msg.str() );
    }

    /**
     * Test for excluded volume by setting all lattice points and count the
     * toal lattice points occupied. If we have overlap this will be smaller
     * than calculated for zero overlap!
     * mPolymerSystem only stores the lower left front corner of the 2x2x2
     * monomer cube. Use that information to set all 8 cells in the lattice
     * to 'occupied'
     */
    /*
     Lattice is an array of size Box_X*Box_Y*Box_Z. PolymerSystem holds the monomer positions which I strongly guess are supposed to be in the range 0<=x<Box_X. If I see correctly, then this part checks for excluded volume by occupying a 2x2x2 cube for each monomer in Lattice and then counting the total occupied cells and compare it to the theoretical value of nMonomers * 8. But Lattice seems to be too small for that kinda usage! I.e. for two particles, one being at x=0 and the other being at x=Box_X-1 this test should return that the excluded volume condition is not met! Therefore the effective box size is actually (Box_X-1,Box_X-1,Box_Z-1) which in my opinion should be a bug ??? */
    std::memset( mLattice, 0, mBoxX * mBoxY * mBoxZ * sizeof( *mLattice ) );
    for ( uint32_t i = 0; i < nAllMonomers; ++i )
    {
        int32_t const & x = mPolymerSystem[ 4*i   ];
        int32_t const & y = mPolymerSystem[ 4*i+1 ];
        int32_t const & z = mPolymerSystem[ 4*i+2 ];
        /**
         * @verbatim
         *           ...+---+---+
         *     ...'''   | 6 | 7 |
         *    +---+---+ +---+---+    y
         *    | 2 | 3 | | 4 | 5 |    ^ z
         *    +---+---+ +---+---+    |/
         *    | 0 | 1 |   ...'''     +--> x
         *    +---+---+'''
         * @endverbatim
         */
        mLattice[ linearizeBoxVectorIndex( x  , y  , z   ) ] = 1; /* 0 */
        mLattice[ linearizeBoxVectorIndex( x+1, y  , z   ) ] = 1; /* 1 */
        mLattice[ linearizeBoxVectorIndex( x  , y+1, z   ) ] = 1; /* 2 */
        mLattice[ linearizeBoxVectorIndex( x+1, y+1, z   ) ] = 1; /* 3 */
        mLattice[ linearizeBoxVectorIndex( x  , y  , z+1 ) ] = 1; /* 4 */
        mLattice[ linearizeBoxVectorIndex( x+1, y  , z+1 ) ] = 1; /* 5 */
        mLattice[ linearizeBoxVectorIndex( x  , y+1, z+1 ) ] = 1; /* 6 */
        mLattice[ linearizeBoxVectorIndex( x+1, y+1, z+1 ) ] = 1; /* 7 */
    }
    /* check total occupied cells inside lattice to ensure that the above
     * transfer went without problems. Note that the number will be smaller
     * if some monomers overlap!
     * Could also simply reduce mLattice with +, I think, because it only
     * cotains 0 or 1 ??? */
    unsigned nOccupied = 0;
    for ( unsigned i = 0u; i < mBoxX * mBoxY * mBoxZ; ++i )
        nOccupied += mLattice[i] != 0;
    if ( ! ( nOccupied == nAllMonomers * 8 ) )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::~checkSystem" << "] "
            << "Occupation count in mLattice is wrong! Expected 8*nMonomers="
            << 8 * nAllMonomers << " occupied cells, but got " << nOccupied;
        throw std::runtime_error( msg.str() );
    }

    /**
     * Check bonds i.e. that |dx|<=3 and whether it is allowed by the given
     * bond set
     */
    for ( unsigned i = 0; i < nAllMonomers; ++i )
    for ( unsigned iNeighbor = 0; iNeighbor < mNeighbors[i].size; ++iNeighbor )
    {
        /* calculate the bond vector between the neighbor and this particle
         * neighbor - particle = ( dx, dy, dz ) */
        intCUDA * const neighbor = & mPolymerSystem[ 4*mNeighbors[i].neighborIds[ iNeighbor ] ];
        int32_t const dx = neighbor[0] - mPolymerSystem[ 4*i+0 ];
        int32_t const dy = neighbor[1] - mPolymerSystem[ 4*i+1 ];
        int32_t const dz = neighbor[2] - mPolymerSystem[ 4*i+2 ];

        int erroneousAxis = -1;
        if ( ! ( -3 <= dx && dx <= 3 ) ) erroneousAxis = 0;
        if ( ! ( -3 <= dy && dy <= 3 ) ) erroneousAxis = 1;
        if ( ! ( -3 <= dz && dz <= 3 ) ) erroneousAxis = 2;
        if ( erroneousAxis >= 0 || mForbiddenBonds[ linearizeBondVectorIndex( dx, dy, dz ) ] )
        {
            std::stringstream msg;
            msg << "[" << __FILENAME__ << "::checkSystem] ";
            if ( erroneousAxis > 0 )
                msg << "Invalid " << 'X' + erroneousAxis << "Bond: ";
            if ( mForbiddenBonds[ linearizeBondVectorIndex( dx, dy, dz ) ] )
                msg << "This particular bond is forbidden: ";
            msg << "(" << dx << "," << dy<< "," << dz << ") between monomer "
                << i+1 << " at (" << mPolymerSystem[ 4*i+0 ] << ","
                                  << mPolymerSystem[ 4*i+1 ] << ","
                                  << mPolymerSystem[ 4*i+2 ] << ") and monomer "
                << mNeighbors[i].neighborIds[ iNeighbor ]+1 << " at ("
                << neighbor[0] << "," << neighbor[1] << "," << neighbor[2] << ")"
                << std::endl;
             throw std::runtime_error( msg.str() );
        }
    }
}

void UpdaterGPUScBFM_AB_Type::runSimulationOnGPU
(
    int32_t const nMonteCarloSteps
)
{
    std::clock_t const t0 = std::clock();

    auto const nSpecies = mnElementsInGroup.size();

    /**
     * Statistics (min, max, mean, stddev) on filtering. Filtered because of:
     *   0: bonds, 1: volume exclusion, 2: volume exclusion (parallel)
     * These statistics are done for each group separately
     */
    std::vector< std::vector< double > > sums, sums2, mins, maxs, ns;
    std::vector< unsigned long long int > vFiltered;
    unsigned long long int * dpFiltered = NULL;
    auto constexpr nFilters = 5;
    if ( mLog.isActive( "Stats" ) )
    {
        sums .resize( nSpecies, std::vector< double >( nFilters, 0            ) );
        sums2.resize( nSpecies, std::vector< double >( nFilters, 0            ) );
        mins .resize( nSpecies, std::vector< double >( nFilters, nAllMonomers ) );
        maxs .resize( nSpecies, std::vector< double >( nFilters, 0            ) );
        ns   .resize( nSpecies, std::vector< double >( nFilters, 0            ) );
        /* ns needed because we need to know how often each group was advanced */
        vFiltered.resize( nFilters );
        CUDA_ERROR( cudaMalloc( &dpFiltered, nFilters * sizeof( *dpFiltered ) ) );
        CUDA_ERROR( cudaMemsetAsync( (void*) dpFiltered, 0, nFilters * sizeof( *dpFiltered ), mStream ) );
    }

    /**
     * Logic for determining the best threadsPerBlock configuration
     *
     * This might be dependent on the species, therefore for each species
     * store the current best thread count and all timings.
     * As the cudaEventSynchronize timings are expensive, stop benchmarking
     * after having found the best configuration.
     * Only try out power two multiples of warpSize up to maxThreadsPerBlock,
     * e.g. 32, 64, 128, 256, 512, 1024, because smaller than warp size
     * should never lead to a speedup and non power of twos, e.g. 196 even,
     * won't be able to perfectly fill out the shared multi processor.
     * Also, automatically determine whether cudaMemset is faster or not (after
     * we found the best threads per block configuration
     * note: test example best configuration was 128 threads per block and use
     *       the cudaMemset version instead of the third kernel
     */
    std::vector< int > vnThreadsToTry;
    for ( auto nThreads = mCudaProps.warpSize; nThreads <= mCudaProps.maxThreadsPerBlock; nThreads *= 2 )
        vnThreadsToTry.push_back( nThreads );
    assert( vnThreadsToTry.size() > 0 );
    struct SpeciesBenchmarkData
    {
        /* 2 vectors of double for measurements with and without cudaMemset */
        std::vector< std::vector< float > > timings;
        std::vector< std::vector< int   > > nRepeatTimings;
        int iBestThreadCount;
        bool useCudaMemset;
        bool decidedAboutThreadCount;
        bool decidedAboutCudaMemset;
    };
    std::vector< SpeciesBenchmarkData > benchmarkInfo( nSpecies, SpeciesBenchmarkData{
        std::vector< std::vector< float > >( 2 /* true or false */,
            std::vector< float >( vnThreadsToTry.size(),
                std::numeric_limits< float >::infinity() ) ),
        std::vector< std::vector< int   > >( 2 /* true or false */,
            std::vector< int   >( vnThreadsToTry.size(),
            2 /* repeat 2 time, i.e. execute three times */ ) ),
#ifdef AUTO_CONFIGURE_BEST_SETTINGS_FOR_PSCBFM_ALGORITHM
        0, true, vnThreadsToTry.size() <= 1, false
#else
        2, true, true, true
#endif
    } );
    cudaEvent_t tOneGpuLoop0, tOneGpuLoop1;
    cudaEventCreate( &tOneGpuLoop0 );
    cudaEventCreate( &tOneGpuLoop1 );

    cudaEvent_t tGpu0, tGpu1;
    if ( mLog.isActive( "Benchmark" ) )
    {
        cudaEventCreate( &tGpu0 );
        cudaEventCreate( &tGpu1 );
        cudaEventRecord( tGpu0, mStream );
    }

    auto const nMonomersPadded = mPolymerSystemSorted->nElements / 3;
    /* run simulation */
    for ( int32_t iStep = 1; iStep <= nMonteCarloSteps; ++iStep )
    {
        /* one Monte-Carlo step:
         *  - tries to move on average all particles one time
         *  - each particle could be touched, not just one group */
        for ( uint32_t iSubStep = 0; iSubStep < nSpecies; ++iSubStep )
        {
            /* randomly choose which monomer group to advance */
            auto const iSpecies = randomNumbers.r250_rand32() % nSpecies;
            auto const seed     = randomNumbers.r250_rand32();
            auto const nThreads = vnThreadsToTry.at( benchmarkInfo[ iSpecies ].iBestThreadCount );
            auto const nBlocks  = ceilDiv( mnElementsInGroup[ iSpecies ], nThreads );
            auto const needToBenchmark = ! (
                benchmarkInfo[ iSpecies ].decidedAboutThreadCount &&
                benchmarkInfo[ iSpecies ].decidedAboutCudaMemset );
            auto const useCudaMemset = benchmarkInfo[ iSpecies ].useCudaMemset;
            if ( needToBenchmark )
                cudaEventRecord( tOneGpuLoop0, mStream );

            /*
            if ( iStep < 3 )
                mLog( "Info" ) << "Calling Check-Kernel for species " << iSpecies << " for uint32_t * " << (void*) mNeighborsSorted->gpu << " + " << mNeighborsSortedInfo.getMatrixOffsetElements( iSpecies ) << " = " << (void*)( mNeighborsSorted->gpu + mNeighborsSortedInfo.getMatrixOffsetElements( iSpecies ) ) << " with pitch " << mNeighborsSortedInfo.getMatrixPitchElements( iSpecies ) << "\n";
            */

            kernelSimulationScBFMCheckSpecies
            <<< nBlocks, nThreads, 0, mStream >>>(
                mPolymerSystemSorted->gpu,
                mPolymerFlags->gpu,
                nMonomersPadded, /* pitch in elements for mPolymerSystemSorted */
                iSubGroupOffset[ iSpecies ],
                mLatticeTmp->gpu,
                mNeighborsSorted->gpu + mNeighborsSortedInfo.getMatrixOffsetElements( iSpecies ),
                mNeighborsSortedInfo.getMatrixPitchElements( iSpecies ),
                mNeighborsSortedSizes->gpu,
                mnElementsInGroup[ iSpecies ], seed,
                mLatticeOut->texture
            );

            if ( mLog.isActive( "Stats" ) )
            {
                kernelCountFilteredCheck
                <<< nBlocks, nThreads, 0, mStream >>>(
                    mPolymerSystemSorted->gpu,
                    mPolymerFlags->gpu,
                    nMonomersPadded, /* pitch in elements for mPolymerSystemSorted */
                    iSubGroupOffset[ iSpecies ],
                    mLatticeTmp->gpu,
                    mNeighborsSorted->gpu + mNeighborsSortedInfo.getMatrixOffsetElements( iSpecies ),
                    mNeighborsSortedInfo.getMatrixPitchElements( iSpecies ),
                    mNeighborsSortedSizes->gpu,
                    mnElementsInGroup[ iSpecies ], seed,
                    mLatticeOut->texture,
                    dpFiltered
                );
            }

            if ( useCudaMemset )
            {
                kernelSimulationScBFMPerformSpeciesAndApply
                <<< nBlocks, nThreads, 0, mStream >>>(
                    mPolymerSystemSorted->gpu + iSubGroupOffset[ iSpecies ],
                    mPolymerFlags->gpu + iSubGroupOffset[ iSpecies ],
                    nMonomersPadded, /* pitch in elements for mPolymerSystemSorted */
                    mLatticeOut->gpu,
                    mnElementsInGroup[ iSpecies ],
                    mLatticeTmp->texture
                );
            }
            else
            {
                kernelSimulationScBFMPerformSpecies
                <<< nBlocks, nThreads, 0, mStream >>>(
                    mPolymerSystemSorted->gpu + iSubGroupOffset[ iSpecies ],
                    mPolymerFlags->gpu + iSubGroupOffset[ iSpecies ],
                    nMonomersPadded, /* pitch in elements for mPolymerSystemSorted */
                    mLatticeOut->gpu,
                    mnElementsInGroup[ iSpecies ],
                    mLatticeTmp->texture
                );
            }

            if ( mLog.isActive( "Stats" ) )
            {
                kernelCountFilteredPerform
                <<< nBlocks, nThreads, 0, mStream >>>(
                    mPolymerSystemSorted->gpu + iSubGroupOffset[ iSpecies ],
                    mPolymerFlags->gpu + iSubGroupOffset[ iSpecies ],
                    nMonomersPadded, /* pitch in elements for mPolymerSystemSorted */
                    mLatticeOut->gpu,
                    mnElementsInGroup[ iSpecies ],
                    mLatticeTmp->texture,
                    dpFiltered
                );
            }

            if ( useCudaMemset )
            {
                #ifdef USE_THRUST_FILL
                    thrust::fill( thrust::system::cuda::par, (uint64_t*)  mLatticeTmp->gpu,
                                  (uint64_t*)( mLatticeTmp->gpu + mLatticeTmp->nElements ), 0 );
                #else
                    #ifdef USE_BIT_PACKING_TMP_LATTICE
                        cudaMemsetAsync( (void*) mLatticeTmp->gpu, 0, mLatticeTmp->nBytes / CHAR_BIT, mStream );
                    #else
                        cudaMemsetAsync( (void*) mLatticeTmp->gpu, 0, mLatticeTmp->nBytes, mStream );
                    #endif
                #endif
            }
            else
            {
                kernelSimulationScBFMZeroArraySpecies
                <<< nBlocks, nThreads, 0, mStream >>>(
                    mPolymerSystemSorted->gpu + iSubGroupOffset[ iSpecies ],
                    mPolymerFlags->gpu + iSubGroupOffset[ iSpecies ],
                    nMonomersPadded, /* pitch in elements for mPolymerSystemSorted */
                    mLatticeTmp->gpu,
                    mnElementsInGroup[ iSpecies ]
                );
            }

            if ( needToBenchmark )
            {
                auto & info = benchmarkInfo[ iSpecies ];
                cudaEventRecord( tOneGpuLoop1, mStream );
                cudaEventSynchronize( tOneGpuLoop1 );
                float milliseconds = 0;
                cudaEventElapsedTime( & milliseconds, tOneGpuLoop0, tOneGpuLoop1 );
                auto const iThreadCount = info.iBestThreadCount;
                auto & oldTiming = info.timings.at( useCudaMemset ).at( iThreadCount );
                oldTiming = std::min( oldTiming, milliseconds );

                mLog( "Info" )
                << "Using " << nThreads << " threads (" << nBlocks << " blocks)"
                << " and using " << ( useCudaMemset ? "cudaMemset" : "kernelZeroArray" )
                << " for species " << iSpecies << " took " << milliseconds << "ms\n";

                auto & nStillToRepeat = info.nRepeatTimings.at( useCudaMemset ).at( iThreadCount );
                if ( nStillToRepeat > 0 )
                    --nStillToRepeat;
                else if ( ! info.decidedAboutThreadCount )
                {
                    /* if not the first timing, then decide whether we got slower,
                     * i.e. whether we found the minimum in the last step and
                     * have to roll back */
                    if ( iThreadCount > 0 )
                    {
                        if ( info.timings.at( useCudaMemset ).at( iThreadCount-1 ) < milliseconds )
                        {
                            --info.iBestThreadCount;
                            info.decidedAboutThreadCount = true;
                        }
                        else
                            ++info.iBestThreadCount;
                    }
                    else
                        ++info.iBestThreadCount;
                    /* if we can't increment anymore, then we are finished */
                    assert( (size_t) info.iBestThreadCount <= vnThreadsToTry.size() );
                    if ( (size_t) info.iBestThreadCount == vnThreadsToTry.size() )
                    {
                        --info.iBestThreadCount;
                        info.decidedAboutThreadCount = true;
                    }
                    if ( info.decidedAboutThreadCount )
                    {
                        /* then in the next term try out changing cudaMemset
                         * version to custom kernel version (or vice-versa) */
                        if ( ! info.decidedAboutCudaMemset )
                            info.useCudaMemset = ! info.useCudaMemset;
                        mLog( "Info" )
                        << "Using " << vnThreadsToTry.at( info.iBestThreadCount )
                        << " threads per block is the fastest for species "
                        << iSpecies << ".\n";
                    }
                }
                else if ( ! info.decidedAboutCudaMemset )
                {
                    info.decidedAboutCudaMemset = true;
                    if ( info.timings.at( ! useCudaMemset ).at( iThreadCount ) < milliseconds )
                        info.useCudaMemset = ! useCudaMemset;
                    if ( info.decidedAboutCudaMemset )
                    {
                        mLog( "Info" )
                        << "Using " << ( info.useCudaMemset ? "cudaMemset" : "kernelZeroArray" )
                        << " is the fastest for species " << iSpecies << ".\n";
                    }
                }
            }

            if ( mLog.isActive( "Stats" ) )
            {
                CUDA_ERROR( cudaMemcpyAsync( (void*) &vFiltered.at(0), (void*) dpFiltered,
                    nFilters * sizeof( *dpFiltered ), cudaMemcpyDeviceToHost, mStream ) );
                CUDA_ERROR( cudaStreamSynchronize( mStream ) );
                CUDA_ERROR( cudaMemsetAsync( (void*) dpFiltered, 0, nFilters * sizeof( *dpFiltered ), mStream ) );

                for ( auto iFilter = 0u; iFilter < nFilters; ++iFilter )
                {
                    double const x = vFiltered.at( iFilter );
                    sums .at( iSpecies ).at( iFilter ) += x;
                    sums2.at( iSpecies ).at( iFilter ) += x*x;
                    mins .at( iSpecies ).at( iFilter ) = std::min( mins.at( iSpecies ).at( iFilter ), x );
                    maxs .at( iSpecies ).at( iFilter ) = std::max( maxs.at( iSpecies ).at( iFilter ), x );
                    ns   .at( iSpecies ).at( iFilter ) += 1;
                }
            }
        } // iSubstep
    } // iStep

    if ( mLog.isActive( "Benchmark" ) )
    {
        // https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc/#disqus_thread
        cudaEventRecord( tGpu1, mStream );
        cudaEventSynchronize( tGpu1 );  // basically a StreamSynchronize
        float milliseconds = 0;
        cudaEventElapsedTime( & milliseconds, tGpu0, tGpu1 );
        mLog( "Benchmark" ) << "tGpuLoop = " << milliseconds / 1000. << "s\n";
    }

    if ( mLog.isActive( "Stats" ) && dpFiltered != NULL )
    {
        CUDA_ERROR( cudaFree( dpFiltered ) );
        mLog( "Stats" ) << "Filter analysis. Format:\n" << "Filter Reason: min | mean +- stddev | max\n";
        std::map< int, std::string > filterNames;
        filterNames[0] = "Box Boundaries";
        filterNames[1] = "Invalid Bonds";
        filterNames[2] = "Volume Exclusion";
        filterNames[3] = "! Invalid Bonds && Volume Exclusion";
        filterNames[4] = "! Invalid Bonds && ! Volume Exclusion && Parallel Volume Exclusion";
        for ( auto iGroup = 0u; iGroup < mnElementsInGroup.size(); ++iGroup )
        {
            mLog( "Stats" ) << "\n=== Group " << char( 'A' + iGroup ) << " (" << mnElementsInGroup[ iGroup ] << ") ===\n";
            for ( auto iFilter = 0u; iFilter < nFilters; ++iFilter )
            {
                double const nRepeats = ns.at( iGroup ).at( iFilter );
                double const mean = sums .at( iGroup ).at( iFilter ) / nRepeats;
                double const sum2 = sums2.at( iGroup ).at( iFilter ) / nRepeats;
                auto const stddev = std::sqrt( nRepeats/(nRepeats-1) * ( sum2 - mean * mean ) );
                auto const & min = mins.at( iGroup ).at( iFilter );
                auto const & max = maxs.at( iGroup ).at( iFilter );
                mLog( "Stats" )
                    << filterNames[iFilter] << ": "
                    << min  << "(" << 100. * min  / mnElementsInGroup[ iGroup ] << "%) | "
                    << mean << "(" << 100. * mean / mnElementsInGroup[ iGroup ] << "%) +- "
                    << stddev << " | "
                    << max  << "(" << 100. * max  / mnElementsInGroup[ iGroup ] << "%)\n";
            }
            if ( sums.at( iGroup ).at(0) != 0 )
                mLog( "Stats" ) << "The value for remeaining particles after first kernel will be wrong if we have non-periodic boundary conditions (todo)!\n";
            auto const nAvgFilteredKernel1 = ( sums.at( iGroup ).at(1) + sums.at( iGroup ).at(3) ) / ns.at( iGroup ).at(3);
            mLog( "Stats" ) << "Remaining after 1st kernel: " << mnElementsInGroup[ iGroup ] - nAvgFilteredKernel1 << "(" << 100. * ( mnElementsInGroup[ iGroup ] - nAvgFilteredKernel1 ) / mnElementsInGroup[ iGroup ] << "%)\n";
            auto const nAvgFilteredKernel2 = sums.at( iGroup ).at(4) / ns.at( iGroup ).at(4);
            auto const percentageMoved = ( mnElementsInGroup[ iGroup ] - nAvgFilteredKernel1 - nAvgFilteredKernel2 ) / mnElementsInGroup[ iGroup ];
            mLog( "Stats" ) << "For parallel collisions it's interesting to give the percentage of sorted particles in relation to whose who can actually still move, not in relation to ALL particles\n"
                << "    Third kernel gets " << mnElementsInGroup[ iGroup ] << " monomers, but first kernel (bonds, box, volume exclusion) already filtered " << nAvgFilteredKernel1 << "(" << 100. * nAvgFilteredKernel1 / mnElementsInGroup[ iGroup ] << "%) which the 2nd kernel has to refilter again (if no stream compaction is used).\n"
                << "    Then from the remaining " << mnElementsInGroup[ iGroup ] - nAvgFilteredKernel1 << "(" << 100. * ( mnElementsInGroup[ iGroup ] - nAvgFilteredKernel1 ) / mnElementsInGroup[ iGroup ] << "%) the 2nd kernel filters out another " << nAvgFilteredKernel2 << " particles which in relation to the particles which actually still could move before is: " << 100. * nAvgFilteredKernel2 / ( mnElementsInGroup[ iGroup ] - nAvgFilteredKernel1 ) << "% and in relation to the total particles: " << 100. * nAvgFilteredKernel2 / mnElementsInGroup[ iGroup ] << "%\n"
                << "    Therefore in total (all three kernels) and on average (multiple salves of three kernels) " << ( mnElementsInGroup[ iGroup ] - nAvgFilteredKernel1 - nAvgFilteredKernel2 ) << "(" << 100. * percentageMoved << "%) particles can be moved per step. I.e. it takes on average " << 1. / percentageMoved << " Monte-Carlo steps per monomer until a monomer actually changes position.\n";
        }
    }

    mtCopyBack0 = std::chrono::high_resolution_clock::now();

    /* all MCS are done- copy information back from GPU to host */
    if ( mLog.isActive( "Check" ) )
    {
        mLatticeTmp->pop( false ); // sync
        size_t nOccupied = 0;
        for ( size_t i = 0u; i < mBoxX * mBoxY * mBoxZ; ++i )
            nOccupied += mLatticeTmp->host[i] != 0;
        if ( nOccupied != 0 )
        {
            std::stringstream msg;
            msg << "latticeTmp occupation (" << nOccupied << ") should be 0! Exiting ...\n";
            throw std::runtime_error( msg.str() );
        }
    }

    /* copy into mPolymerSystem and drop the property tag while doing so.
     * would be easier and probably more efficient if mPolymerSystem->gpu/host
     * would be a struct of arrays instead of an array of structs !!! */
    mPolymerSystemSorted->pop( false ); // sync

    if ( mLog.isActive( "Benchmark" ) )
    {
        std::clock_t const t1 = std::clock();
        double const dt = float(t1-t0) / CLOCKS_PER_SEC;
        mLog( "Benchmark" )
        << "run time (GPU): " << nMonteCarloSteps << "\n"
        << "mcs = " << nMonteCarloSteps  << "  speed [performed monomer try and move/s] = MCS*N/t: "
        << nMonteCarloSteps * ( nAllMonomers / dt )  << "     runtime[s]:" << dt << "\n";
    }

    /* untangle reordered array so that LeMonADE can use it again */
    for ( size_t i = 0u; i < nAllMonomers; ++i )
    {
        if ( i < 10 )
            mLog( "Info" ) << "Copying back " << i << " from " << iToiNew[i] << "\n";
        mPolymerSystem[ 4*i+0 ] = mPolymerSystemSorted->host[ 0 * nMonomersPadded + iToiNew[i] ];
        mPolymerSystem[ 4*i+1 ] = mPolymerSystemSorted->host[ 1 * nMonomersPadded + iToiNew[i] ];
        mPolymerSystem[ 4*i+2 ] = mPolymerSystemSorted->host[ 2 * nMonomersPadded + iToiNew[i] ];
    }

    checkSystem(); // no-op if "Check"-level deactivated
}

/**
 * GPUScBFM_AB_Type::initialize and run and cleanup should be usable on
 * repeat. Which means we need to destruct everything created in
 * GPUScBFM_AB_Type::initialize, which encompasses setLatticeSize,
 * setNrOfAllMonomers and initialize. Currently this includes all allocs,
 * so we can simply call destruct.
 */
void UpdaterGPUScBFM_AB_Type::cleanup()
{
    /* check whether connectivities on GPU got corrupted */
    for ( uint32_t i = 0; i < nAllMonomers; ++i )
    {
        unsigned const nNeighbors = ( mPolymerSystem[ 4*i+3 ] & 224 /* 0b11100000 */ ) >> 5;
        if ( nNeighbors != mNeighbors[i].size )
        {
            std::stringstream msg;
            msg << "[" << __FILENAME__ << "::~cleanup" << "] "
                << "Connectivities in property field of mPolymerSystem are "
                << "different from host-side connectivities. This should not "
                << "happen! (Monomer " << i << ": " << nNeighbors << " != "
                << mNeighbors[i].size << "\n";
            throw std::runtime_error( msg.str() );
        }
    }
    this->destruct();

    cudaDeviceSynchronize();
    cudaProfilerStop();
}
