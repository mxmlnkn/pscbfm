/*
 * UpdaterGPUScBFM_AB_Type.cpp
 *
 *  Created on: 27.07.2017
 *      Authors: Ron Dockhorn, Maximilian Knespel
 */

#include "UpdaterGPUScBFM_AB_Type.h"


#include <algorithm>                        // fill, sort, count
#include <chrono>                           // std::chrono::high_resolution_clock
#include <cstdio>                           // printf
#include <cstdlib>                          // exit
#include <cstring>                          // memset
#include <ctime>
#include <functional>
#include <iostream>
#include <limits>                           // numeric_limits
#include <stdexcept>
#include <stdint.h>
#include <sstream>

#include <cuda_profiler_api.h>              // cudaProfilerStop
#ifdef USE_THRUST_FILL
#   include <thrust/system/cuda/execution_policy.h>
#   include <thrust/fill.h>
#endif
#include <thrust/execution_policy.h>        // thrust::seq, thrust::host
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/sort.h>                    // sort_by_key

#include "Fundamental/BitsCompileTime.hpp"
#include "cudacommon.hpp"
#include "SelectiveLogger.hpp"
#include "graphColoring.tpp"


#define DEBUG_UPDATERGPUSCBFM_AB_TYPE 100
#if defined( USE_BIT_PACKING_TMP_LATTICE ) || defined( USE_BIT_PACKING_LATTICE )
#   define USE_BIT_PACKING
#endif



/**
 * anonymous namespace containing the kernels and device functions
 * in order to avoid multiple definition errors as they are only used
 * in this file and are not to be exported!
 */
namespace {


/* shorten full type names for kernels */
using T_Flags            = UpdaterGPUScBFM_AB_Type::T_Flags           ;
using T_Lattice          = UpdaterGPUScBFM_AB_Type::T_Lattice         ;
using vecIntCUDA         = UpdaterGPUScBFM_AB_Type::T_CoordinatesCuda ;
using T_Coordinate       = UpdaterGPUScBFM_AB_Type::T_Coordinate      ;
using T_CoordinateCuda   = UpdaterGPUScBFM_AB_Type::T_CoordinateCuda  ;
using T_Coordinates      = UpdaterGPUScBFM_AB_Type::T_Coordinates     ;
using T_CoordinateCuda   = UpdaterGPUScBFM_AB_Type::T_CoordinateCuda  ;
using T_UCoordinateCuda  = UpdaterGPUScBFM_AB_Type::T_UCoordinateCuda ;
using T_UCoordinatesCuda = UpdaterGPUScBFM_AB_Type::T_UCoordinatesCuda;
using T_Id               = UpdaterGPUScBFM_AB_Type::T_Id              ;
using vecUIntCUDA        = CudaVec4< T_UCoordinateCuda >::value_type  ;


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
__device__ __constant__ uint32_t DXTable2_d[6]; //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z
__device__ __constant__ uint32_t DYTable2_d[6]; //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z
__device__ __constant__ uint32_t DZTable2_d[6]; //0:-x; 1:+x; 2:-y; 3:+y; 4:-z; 5+z
/**
 * If intCUDA is different from uint32_t, then this second table prevents
 * expensive type conversions, but both tables are still needed, the
 * uint32_t version, because the calculation of the linear index will result
 * in uint32_t anyway and the intCUDA version for solely updating the
 * position information
 */
__device__ __constant__ T_CoordinateCuda DXTableIntCUDA_d[6];
__device__ __constant__ T_CoordinateCuda DYTableIntCUDA_d[6];
__device__ __constant__ T_CoordinateCuda DZTableIntCUDA_d[6];

/* will this really bring performance improvement? At least constant cache
 * might be as fast as register access when all threads in a warp access the
 * the same constant */
__device__ __constant__ uint32_t dcBoxX     ;  // lattice size in X
__device__ __constant__ uint32_t dcBoxY     ;  // lattice size in Y
__device__ __constant__ uint32_t dcBoxZ     ;  // lattice size in Z
__device__ __constant__ uint32_t dcBoxXM1   ;  // lattice size in X-1
__device__ __constant__ uint32_t dcBoxYM1   ;  // lattice size in Y-1
__device__ __constant__ uint32_t dcBoxZM1   ;  // lattice size in Z-1
__device__ __constant__ uint32_t dcBoxXLog2 ;  // lattice shift in X
__device__ __constant__ uint32_t dcBoxXYLog2;  // lattice shift in X*Y


}


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

__device__ inline uint32_t hash( uint32_t a )
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
    x = ( x + ( x << ( iStep2Pow * nSpacing ) ) ) & mask;
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

template< typename T >
__device__ __host__ inline bool isPowerOfTwo( T const & x )
{
    //return ! ( x == T(0) ) && ! ( x & ( x - T(1) ) );
    return __popc( x ) <= 1;
}

/**
 * @see https://en.wikipedia.org/wiki/Moore_curve
 * @see http://www.grant-trebbin.com/2017/03/calculating-hilbert-curve-coordinates.html
 * @see "Programming the Hilbert curve - John Skilling - 2004"
 * @see https://testbook.com/blog/conversion-from-gray-code-to-binary-code-and-vice-versa/
 * @see https://rosettacode.org/wiki/Gray_code#C.2B.2B
 * @see https://stackoverflow.com/questions/17490431/gray-code-increment-function
 * @see https://en.wikipedia.org/wiki/Gray_code#Converting_to_and_from_Gray_code
 */
template< typename T >
__device__ __host__ inline T toGrayCode( T x )
{
    return x ^ ( x >> 1 );
}

template< typename T >
__device__ __host__ inline T fromGrayCode( T x )
{
    T p = x;
    while ( x >>= 1 )
        p ^= x;
    return p;
}

/**
 * same as above, but instead of mBoxXM1 it uses dcBoxXM1,
 * which resides in constant memory
 * Input coordinates are implicitly converted to T_Id on call assuming
 * that T_Id is generally larger than T_Coordinate
 */
__device__ inline T_Id linearizeBoxVectorIndex
(
    uint32_t const & ix,
    uint32_t const & iy,
    uint32_t const & iz
)
{
    #if defined ( USE_ZCURVE_FOR_LATTICE ) || defined ( USE_MOORE_CURVE_FOR_LATTICE )
        auto const zorder =
              diluteBits< T_Id, 2 >( ix & dcBoxXM1 )        +
            ( diluteBits< T_Id, 2 >( iy & dcBoxYM1 ) << 1 ) +
            ( diluteBits< T_Id, 2 >( iz & dcBoxZM1 ) << 2 );
        #if defined ( USE_MOORE_CURVE_FOR_LATTICE )
            return fromGrayCode( zorder );
        #else
            return zorder;
        #endif
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

#ifdef USE_BIT_PACKING
    template< typename T, typename T_Id > __device__ __host__ inline
    T bitPackedGet( T const * const & p, T_Id const & i )
    {
        /**
         * >> 3, because 3 bits = 2^3=8 numbers are used for sub-byte indexing,
         * i.e. we divide the index i by 8 which is equal to the space we save
         * by bitpacking.
         * & 7, because 7 = 0b111, i.e. we are only interested in the last 3
         * bits specifying which subbyte element we want
         */
        return ( p[ i >> 3 ] >> ( i & T_Id(7) ) ) & T(1);
    }

    template< typename T > __device__ inline
    T bitPackedTextureGet( cudaTextureObject_t p, int i )
    {
        return ( tex1Dfetch<T>( p, i >> 3 ) >> ( i & 7 ) ) & T(1);
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
    template< typename T, typename T_Id > __device__ __host__ inline
    void bitPackedSet( T * const __restrict__ p, T_Id const & i )
    {
        static_assert( sizeof(int) == 4, "" );
        #ifdef __CUDA_ARCH__
            atomicOr ( (int*) p + ( i >> 5 ),    T(1) << ( i & T_Id( 0x1F ) )   );
        #else
            p[ i >> 3 ] |= T(1) << ( i & T_Id(7) );
        #endif
    }

    template< typename T, typename T_Id > __device__ __host__ inline
    void bitPackedUnset( T * const __restrict__ p, T_Id const & i )
    {
        #ifdef __CUDA_ARCH__
            atomicAnd( (uint32_t*) p + ( i >> 5 ), ~( uint32_t(1) << ( i & T_Id( 0x1F ) ) ) );
        #else
            p[ i >> 3 ] &= ~( T(1) << ( i & T_Id(7) ) );
        #endif
    }
#else
    template< typename T, typename T_Id > __device__ __host__ inline
    T bitPackedGet( T const * const & p, T_Id const & i ){ return p[i]; }
    template< typename T > __device__ inline
    T bitPackedTextureGet( cudaTextureObject_t p, int i ) {
        return tex1Dfetch<T>(p,i); }
    template< typename T, typename T_Id > __device__ __host__ inline
    void bitPackedSet  ( T * const __restrict__ p, T_Id const & i ){ p[i] = 1; }
    template< typename T, typename T_Id > __device__ __host__ inline
    void bitPackedUnset( T * const __restrict__ p, T_Id const & i ){ p[i] = 0; }
#endif


int constexpr iFetchOrder0 = 0;
int constexpr iFetchOrder1 = 3;
int constexpr iFetchOrder2 = 6;
int constexpr iFetchOrder3 = 2;
int constexpr iFetchOrder4 = 5;
int constexpr iFetchOrder5 = 8;
int constexpr iFetchOrder6 = 1;
int constexpr iFetchOrder7 = 4;
int constexpr iFetchOrder8 = 7;


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
    T_Flags             const & axis      ,
    T_Lattice (*fetch)( cudaTextureObject_t, int ) = &tex1Dfetch< T_Lattice >,
    T_Id              * const   iOldPos = NULL
)
{
#if 0 // defined( NOMAGIC )
    if ( iOldPos != NULL )
        *iOldPos =  linearizeBoxVectorIndex( x0, y0, z0 );

    bool isOccupied = false;
    auto const shift = 4*(axis & 1)-2;
    switch ( axis >> 1 )
    {
        #define TMP_FETCH( x,y,z ) \
            (*fetch)( texLattice, linearizeBoxVectorIndex(x,y,z) )
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
    return isOccupied;
#else
    #if defined( USE_ZCURVE_FOR_LATTICE ) || defined ( USE_MOORE_CURVE_FOR_LATTICE )
        auto const x0MDX  = diluteBits< uint32_t, 2 >( ( x0 - uint32_t(1) ) & dcBoxXM1 );
        auto const x0Abs  = diluteBits< uint32_t, 2 >( ( x0               ) & dcBoxXM1 );
        auto const x0PDX  = diluteBits< uint32_t, 2 >( ( x0 + uint32_t(1) ) & dcBoxXM1 );
        auto const y0MDY  = diluteBits< uint32_t, 2 >( ( y0 - uint32_t(1) ) & dcBoxYM1 ) << 1;
        auto const y0Abs  = diluteBits< uint32_t, 2 >( ( y0               ) & dcBoxYM1 ) << 1;
        auto const y0PDY  = diluteBits< uint32_t, 2 >( ( y0 + uint32_t(1) ) & dcBoxYM1 ) << 1;
        auto const z0MDZ  = diluteBits< uint32_t, 2 >( ( z0 - uint32_t(1) ) & dcBoxZM1 ) << 2;
        auto const z0Abs  = diluteBits< uint32_t, 2 >( ( z0               ) & dcBoxZM1 ) << 2;
        auto const z0PDZ  = diluteBits< uint32_t, 2 >( ( z0 + uint32_t(1) ) & dcBoxZM1 ) << 2;
    #else
        auto const x0MDX  =   ( x0 - uint32_t(1) ) & dcBoxXM1;
        auto const x0Abs  =   ( x0               ) & dcBoxXM1;
        auto const x0PDX  =   ( x0 + uint32_t(1) ) & dcBoxXM1;
        auto const y0MDY  = ( ( y0 - uint32_t(1) ) & dcBoxYM1 ) << dcBoxXLog2;
        auto const y0Abs  = ( ( y0               ) & dcBoxYM1 ) << dcBoxXLog2;
        auto const y0PDY  = ( ( y0 + uint32_t(1) ) & dcBoxYM1 ) << dcBoxXLog2;
        auto const z0MDZ  = ( ( z0 - uint32_t(1) ) & dcBoxZM1 ) << dcBoxXYLog2;
        auto const z0Abs  = ( ( z0               ) & dcBoxZM1 ) << dcBoxXYLog2;
        auto const z0PDZ  = ( ( z0 + uint32_t(1) ) & dcBoxZM1 ) << dcBoxXYLog2;
    #endif

    auto const dx = DXTable_d[ axis ];   // 2*axis-1
    auto const dy = DYTable_d[ axis ];   // 2*(axis&1)-1
    auto const dz = DZTable_d[ axis ];   // 2*(axis&1)-1

    if ( iOldPos != NULL )
        *iOldPos = x0Abs + y0Abs + z0Abs;

    uint32_t is[9];

    #if defined( USE_ZCURVE_FOR_LATTICE ) || defined ( USE_MOORE_CURVE_FOR_LATTICE )
        switch ( axis >> 1 )
        {
            case 0: is[7] = ( x0 + dx + dx ) & dcBoxXM1; break;
            case 1: is[7] = ( y0 + dy + dy ) & dcBoxYM1; break;
            case 2: is[7] = ( z0 + dz + dz ) & dcBoxZM1; break;
        }
        is[7] = diluteBits< uint32_t, 2 >( is[7] ) << ( axis >> 1 );
    #else
        switch ( axis >> 1 )
        {
            case 0: is[7] =   ( x0 + 2*dx ) & dcBoxXM1; break;
            case 1: is[7] = ( ( y0 + 2+dy ) & dcBoxYM1 ) << dcBoxXLog2; break;
            case 2: is[7] = ( ( z0 + 2*dz ) & dcBoxZM1 ) << dcBoxXYLog2; break;
        }
    #endif

#define CHECK_FRONT_BIT_PACKED_INDEX_CALC_VERSION 0

#if CHECK_FRONT_BIT_PACKED_INDEX_CALC_VERSION == 6 // same as version 5, but with preemptive return like version 1
    is[8] = is[7];
    auto const direction = axis >> 1;
    auto const isX = ( axis & 6 ) == 0;
    is[6] = isX ? y0MDY : x0MDX;
    is[7] = isX ? y0Abs : x0Abs;
    auto const b0p1 = ( axis & 6 ) == 0 ? y0PDY : x0PDX;
    if ( direction == 2 ) {
        is[2] = is[8] + y0MDY; is[5] = is[8] + y0Abs; is[8] += y0PDY;
    } else {
        is[2] = is[8] + z0MDZ; is[5] = is[8] + z0Abs; is[8] += z0PDZ;
    }
    is[0]  = is[2] + is[6];
    is[3]  = is[5] + is[6];
    is[6] += is[8];
    is[1]  = is[2] + is[7];

    #if defined( USE_MOORE_CURVE_FOR_LATTICE )
        is[0] = fromGrayCode ( is[0] );
        is[3] = fromGrayCode ( is[3] );
        is[6] = fromGrayCode ( is[6] );
        is[1] = fromGrayCode ( is[1] );
    #endif

    if ( ( (*fetch)( texLattice, is[0] ) +
           (*fetch)( texLattice, is[3] ) +
           (*fetch)( texLattice, is[6] ) +
           (*fetch)( texLattice, is[1] ) ) )
        return true;

    is[4]  = is[5] + is[7];
    is[7] += is[8];
    is[2] += b0p1;
    is[5] += b0p1;
    is[8] += b0p1;

    #if defined( USE_MOORE_CURVE_FOR_LATTICE )
        is[2] = fromGrayCode( is[2] );
        is[5] = fromGrayCode( is[5] );
        is[8] = fromGrayCode( is[8] );
        is[4] = fromGrayCode( is[4] );
        is[7] = fromGrayCode( is[7] );
    #endif

    return (*fetch)( texLattice, is[2] ) +
           (*fetch)( texLattice, is[5] ) +
           (*fetch)( texLattice, is[8] ) +
           (*fetch)( texLattice, is[4] ) +
           (*fetch)( texLattice, is[7] );
#elif CHECK_FRONT_BIT_PACKED_INDEX_CALC_VERSION == 5 // try to reduce registers and times even mroe
    is[8] = is[7];
    auto const direction = axis >> 1;
    auto const isX = ( axis & 6 ) == 0;
    is[6] = isX ? y0MDY : x0MDX;
    is[7] = isX ? y0Abs : x0Abs;
    auto const b0p1 = ( axis & 6 ) == 0 ? y0PDY : x0PDX;
    if ( direction == 2 ) {
        is[2] = is[8] + y0MDY; is[5] = is[8] + y0Abs; is[8] += y0PDY;
    } else {
        is[2] = is[8] + z0MDZ; is[5] = is[8] + z0Abs; is[8] += z0PDZ;
    }
    is[0]  = is[2] + is[6];
    is[3]  = is[5] + is[6];
    is[6] += is[8];
    is[1]  = is[2] + is[7];
    is[4]  = is[5] + is[7];
    is[7] += is[8];
    is[2] += b0p1;
    is[5] += b0p1;
    is[8] += b0p1;
#elif CHECK_FRONT_BIT_PACKED_INDEX_CALC_VERSION == 4 // ternary operator version
    auto const direction = axis >> 1;
    auto const a0m1 = direction == 2 ? y0MDY : z0MDZ;
    auto const a0   = direction == 2 ? y0Abs : z0Abs;
    auto const a0p1 = direction == 2 ? y0PDY : z0PDZ;
    auto const b0m1 = direction == 0 ? y0MDY : x0MDX;
    auto const b0   = direction == 0 ? y0Abs : x0Abs;
    auto const b0p1 = direction == 0 ? y0PDY : x0PDX;
    is[2] = is[7] + a0m1; is[5] = is[7] + a0; is[8]  = is[7] + a0p1;
    is[0] = is[2] + b0m1; is[1] = is[2] + b0; is[2] += b0p1;
    is[3] = is[5] + b0m1; is[4] = is[5] + b0; is[5] += b0p1;
    is[6] = is[8] + b0m1; is[7] = is[8] + b0; is[8] += b0p1;
#elif CHECK_FRONT_BIT_PACKED_INDEX_CALC_VERSION == 3 // this version tries to do the calculations unbranched and reorders the calculations in order to hopefully use less registers
    uint32_t a0m1,a0,a0p1;
    if ( axis >> 1 == 2 )
    {
        a0m1 = y0MDY;
        a0   = y0Abs;
        a0p1 = y0PDY;
    }
    else
    {
        a0m1 = z0MDZ;
        a0   = z0Abs;
        a0p1 = z0PDZ;
    }
    is[2] = is[7] + a0m1; is[5]  = is[7] + a0; is[8]  = is[7] + a0p1;
    uint32_t b0m1,b0,b0p1;
    if ( axis >> 1 == 0 )
    {
        b0m1 = y0MDY;
        b0   = y0Abs;
        b0p1 = y0PDY;
    }
    else
    {
        b0m1 = x0MDX;
        b0   = x0Abs;
        b0p1 = x0PDX;
    }
    is[0] = is[2] + b0m1; is[1] = is[2] + b0;
    is[3] = is[5] + b0m1; is[4] = is[5] + b0;
    is[6] = is[8] + b0m1; is[7] = is[8] + b0;
    is[2] += b0p1;
    is[5] += b0p1;
    is[8] += b0p1;
#elif CHECK_FRONT_BIT_PACKED_INDEX_CALC_VERSION == 2 // version which at least tries to reduce the large switch to two smaller if-else's
    if ( axis >> 1 == 2 )
    {
        is[2] = is[7] + y0MDY; is[5] = is[7] + y0Abs; is[8] = is[7] + y0PDY;
    }
    else
    {
        is[2] = is[7] + z0MDZ; is[5] = is[7] + z0Abs; is[8] = is[7] + z0PDZ;
    }
    if ( axis >> 1 == 0 )
    {
        is[0] = is[2] + y0MDY; is[1]  = is[2] + y0Abs;
        is[3] = is[5] + y0MDY; is[4]  = is[5] + y0Abs;
        is[6] = is[8] + y0MDY; is[7]  = is[8] + y0Abs;
        is[2] += y0PDY;
        is[5] += y0PDY;
        is[8] += y0PDY;
    }
    else
    {
        is[0] = is[2] + x0MDX; is[1]  = is[2] + x0Abs;
        is[3] = is[5] + x0MDX; is[4]  = is[5] + x0Abs;
        is[6] = is[8] + x0MDX; is[7]  = is[8] + x0Abs;
        is[2] += x0PDX;
        is[5] += x0PDX;
        is[8] += x0PDX;
    }
#elif CHECK_FRONT_BIT_PACKED_INDEX_CALC_VERSION == 1 // this version tries to implement a preemptive return
    switch ( axis >> 1 )
    {
        case 0: //-+x
            is[2]  = is[7] + z0MDZ; is[5]  = is[7] + z0Abs; is[8]  = is[7] + z0PDZ;
            is[0]  = is[2] + y0MDY;
            is[3]  = is[5] + y0MDY;
            is[6]  = is[8] + y0MDY;
            break;
        case 1: //-+y
            is[2]  = is[7] + z0MDZ; is[5]  = is[7] + z0Abs; is[8]  = is[7] + z0PDZ;
            is[0]  = is[2] + x0MDX;
            is[3]  = is[5] + x0MDX;
            is[6]  = is[8] + x0MDX;
            break;
        case 2: //-+z
            is[2]  = is[7] + y0MDY; is[5]  = is[7] + y0Abs; is[8]  = is[7] + y0PDY;
            is[0]  = is[2] + x0MDX;
            is[3]  = is[5] + x0MDX;
            is[6]  = is[8] + x0MDX;
            break;
    }

    #if defined( USE_MOORE_CURVE_FOR_LATTICE )
        is[0] = fromGrayCode( is[0] );
        is[3] = fromGrayCode( is[3] );
        is[6] = fromGrayCode( is[6] );
    #endif

    if ( ( (*fetch)( texLattice, is[0] ) +
           (*fetch)( texLattice, is[3] ) +
           (*fetch)( texLattice, is[6] ) ) )
        return true;

    switch ( axis >> 1 )
    {
        case 0: //-+x
            is[1]  = is[2] + y0Abs; is[2] += y0PDY;
            is[4]  = is[5] + y0Abs; is[5] += y0PDY;
            is[7]  = is[8] + y0Abs; is[8] += y0PDY;
            break;
        case 1: //-+y
            is[1]  = is[2] + x0Abs; is[2] += x0PDX;
            is[4]  = is[5] + x0Abs; is[5] += x0PDX;
            is[7]  = is[8] + x0Abs; is[8] += x0PDX;
            break;
        case 2: //-+z
            is[1]  = is[2] + x0Abs; is[2] += x0PDX;
            is[4]  = is[5] + x0Abs; is[5] += x0PDX;
            is[7]  = is[8] + x0Abs; is[8] += x0PDX;
            break;
    }

    #if defined( USE_MOORE_CURVE_FOR_LATTICE )
        is[1] = fromGrayCode( is[1] );
        is[2] = fromGrayCode( is[2] );
        is[4] = fromGrayCode( is[4] );
        is[5] = fromGrayCode( is[5] );
        is[7] = fromGrayCode( is[7] );
        is[8] = fromGrayCode( is[8] );
    #endif

    return (*fetch)( texLattice, is[2] ) +
           (*fetch)( texLattice, is[5] ) +
           (*fetch)( texLattice, is[8] ) +
           (*fetch)( texLattice, is[1] ) +
           (*fetch)( texLattice, is[4] ) +
           (*fetch)( texLattice, is[7] );
#elif CHECK_FRONT_BIT_PACKED_INDEX_CALC_VERSION == 0
    switch ( axis >> 1 )
    {
        case 0: //-+x
            /* this line adds all three z directions */
            is[2]  = is[7] + z0MDZ; is[5]  = is[7] + z0Abs; is[8]  = is[7] + z0PDZ;
            /* now for all three results we generate all 3 different y positions */
            is[0]  = is[2] + y0MDY; is[1]  = is[2] + y0Abs; is[2] +=         y0PDY;
            is[3]  = is[5] + y0MDY; is[4]  = is[5] + y0Abs; is[5] +=         y0PDY;
            is[6]  = is[8] + y0MDY; is[7]  = is[8] + y0Abs; is[8] +=         y0PDY;
            break;
            /**
             * so the order for the 9 positions when moving in x direction is:
             * @verbatim
             * z ^
             *   | 0 1 2
             *   | 3 4 5
             *   | 6 7 8
             *   +------> y
             * @endverbatim
             */
        case 1: //-+y
            is[2]  = is[7] + z0MDZ; is[5]  = is[7] + z0Abs; is[8]  = is[7] + z0PDZ;
            is[0]  = is[2] + x0MDX; is[1]  = is[2] + x0Abs; is[2] +=         x0PDX;
            is[3]  = is[5] + x0MDX; is[4]  = is[5] + x0Abs; is[5] +=         x0PDX;
            is[6]  = is[8] + x0MDX; is[7]  = is[8] + x0Abs; is[8] +=         x0PDX;
            break;
            /**
             * @verbatim
             * z ^
             *   | 0 1 2
             *   | 3 4 5
             *   | 6 7 8
             *   +------> x
             * @endverbatim
             */
        case 2: //-+z
            is[2]  = is[7] + y0MDY; is[5]  = is[7] + y0Abs; is[8]  = is[7] + y0PDY;
            is[0]  = is[2] + x0MDX; is[1]  = is[2] + x0Abs; is[2] +=         x0PDX;
            is[3]  = is[5] + x0MDX; is[4]  = is[5] + x0Abs; is[5] +=         x0PDX;
            is[6]  = is[8] + x0MDX; is[7]  = is[8] + x0Abs; is[8] +=         x0PDX;
            break;
            /**
             * @verbatim
             * y ^
             *   | 0 1 2
             *   | 3 4 5
             *   | 6 7 8
             *   +------> x
             * @endverbatim
             */
    }
#endif
    /**
     * we might be able to profit from remporal caching by changing the fetch
     * order ?! In that case the best should be in order of the z-curve.
     * E.g. for a full 2x2x2 unit cell we have the order: A kind of problem is:
     * we want to check a 3x3x1 front ... I don't think there is any easy way
     * to derive a "correct" or "best order ... need to test all. There are
     * 9! = 362880 possibilities ... ... ... I can't recompile all this every
     * time, might have to to some templating and then loop ofer it but that
     * could lead to code bloat ...
     * First, manually try out if changing the order does change any thing in
     * the first place ...
     * The z curve in 3D is:
     * @verbatim
     *   i -> bin  -> (z,y,x)
     *   0 -> 000b -> (0,0,0)
     *   1 -> 001b -> (0,0,1)
     *   2 -> 010b -> (0,1,0)
     *   3 -> 011b -> (0,1,1)
     *   4 -> 100b -> (1,0,0)
     *   5 -> 101b -> (1,0,1)
     *   6 -> 110b -> (1,1,0)
     *   7 -> 111b -> (1,1,1)
     *
     *
     *       .'+---+---+
     *     .'  | 0 | 1 |       y
     *   .'    +---+---+       ^
     *  +---+---+2 | 3 |       |
     *  | 4 | 5 |--+---+       +--> x
     *  +---+---+    .'      .'
     *  | 6 | 7 |  .'       L z
     *  +---+---+.'
     * @endverbatim
     */
#if ( CHECK_FRONT_BIT_PACKED_INDEX_CALC_VERSION != 1 ) && ( CHECK_FRONT_BIT_PACKED_INDEX_CALC_VERSION != 6 )

    #if defined( USE_MOORE_CURVE_FOR_LATTICE )
        is[0] = fromGrayCode( is[0] );
        is[1] = fromGrayCode( is[1] );
        is[2] = fromGrayCode( is[2] );
        is[3] = fromGrayCode( is[3] );
        is[4] = fromGrayCode( is[4] );
        is[5] = fromGrayCode( is[5] );
        is[6] = fromGrayCode( is[6] );
        is[7] = fromGrayCode( is[7] );
        is[8] = fromGrayCode( is[8] );
    #endif

    return (*fetch)( texLattice, is[ iFetchOrder0 ] ) +
           (*fetch)( texLattice, is[ iFetchOrder1 ] ) +
           (*fetch)( texLattice, is[ iFetchOrder2 ] ) +
           (*fetch)( texLattice, is[ iFetchOrder3 ] ) +
           (*fetch)( texLattice, is[ iFetchOrder4 ] ) +
           (*fetch)( texLattice, is[ iFetchOrder5 ] ) +
           (*fetch)( texLattice, is[ iFetchOrder6 ] ) +
           (*fetch)( texLattice, is[ iFetchOrder7 ] ) +
           (*fetch)( texLattice, is[ iFetchOrder8 ] );
#endif
#endif // NOMAGIC
}

__device__ __host__ inline int16_t linearizeBondVectorIndex
(
    int16_t const x,
    int16_t const y,
    int16_t const z
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
    return   ( x & int16_t(7) /* 0b111 */ ) +
           ( ( y & int16_t(7) /* 0b111 */ ) << 3 ) +
           ( ( z & int16_t(7) /* 0b111 */ ) << 6 );
}


namespace {


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
__global__ void kernelSimulationScBFMCheckSpecies
(
    vecIntCUDA  const * const __restrict__ dpPolymerSystem         ,
    T_Flags           * const              dpPolymerFlags          ,
    uint32_t            const              iOffset                 ,
    T_Lattice         * const __restrict__ dpLatticeTmp            ,
    T_Id        const * const              dpNeighbors             ,
    uint32_t            const              rNeighborsPitchElements ,
    uint8_t     const * const              dpNeighborsSizes        ,
    T_Id                const              nMonomers               ,
    uint32_t            const              rSeed                   ,
    cudaTextureObject_t const              texLatticeRefOut
)
{
    uint32_t rn;
    int iGrid = 0;
    for ( auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x, ++iGrid )
    {
        auto const r0 = dpPolymerSystem[ iOffset + iMonomer ];
        /* upcast int3 to int4 in preparation to use PTX SIMD instructions */
        //int4 const r0 = { r0Raw.x, r0Raw.y, r0Raw.z, 0 }; // not faster nor slower
        //select random direction. Own implementation of an rng :S? But I think it at least# was initialized using the LeMonADE RNG ...
        if ( iGrid % 1 == 0 ) // 12 = floor( log(2^32) / log(6) )
            rn = hash( hash( iMonomer ) ^ rSeed );
        uint32_t const direction = rn % uint32_t(6); rn /= uint32_t(6);
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
        if ( T_CoordinateCuda(0) <= r1.x && r1.x < dcBoxXM1 &&
             T_CoordinateCuda(0) <= r1.y && r1.y < dcBoxYM1 &&
             T_CoordinateCuda(0) <= r1.z && r1.z < dcBoxZM1    )
        {
    #endif
            /* check whether the new position would result in invalid bonds
             * between this monomer and its neighbors */
            auto const nNeighbors = dpNeighborsSizes[ iMonomer ];
            bool forbiddenBond = false;
            for ( auto iNeighbor = decltype( nNeighbors )(0); iNeighbor < nNeighbors; ++iNeighbor )
            {
                auto const iGlobalNeighbor = dpNeighbors[ iNeighbor * rNeighborsPitchElements + iMonomer ];
                auto const data2 = dpPolymerSystem[ iGlobalNeighbor ];
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
                properties = ( direction << 2 ) + T_Flags(1) /* can-move-flag */;
            #ifdef USE_BIT_PACKING_TMP_LATTICE
                bitPackedSet( dpLatticeTmp, linearizeBoxVectorIndex( r1.x, r1.y, r1.z ) );
            #else
                dpLatticeTmp[ linearizeBoxVectorIndex( r1.x, r1.y, r1.z ) ] = 1;
            #endif
            }
    #ifdef NONPERIODICITY
        }
    #endif
        dpPolymerFlags[ iMonomer ] = properties;
    }
}


__global__ void kernelCountFilteredCheck
(
    vecIntCUDA       const * const __restrict__ dpPolymerSystem        ,
    T_Flags          const * const              dpPolymerFlags         ,
    uint32_t                 const              iOffset                ,
    T_Lattice        const * const __restrict__ /* dpLatticeTmp */     ,
    T_Id             const * const              dpNeighbors            ,
    uint32_t                 const              rNeighborsPitchElements,
    uint8_t          const * const              dpNeighborsSizes       ,
    T_Id                     const              nMonomers              ,
    uint32_t                 const              rSeed                  ,
    cudaTextureObject_t      const              texLatticeRefOut       ,
    unsigned long long int * const              dpFiltered
)
{
    uint32_t rn;
    int iGrid = 0;
    for ( auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x, ++iGrid )
    {
        auto const r0 = dpPolymerSystem[ iOffset + iMonomer ];
        if ( iGrid % 1 == 0 )
            rn = hash( hash( iMonomer ) ^ rSeed );
        uint32_t const direction = rn % uint32_t(6); rn /= uint32_t(6);

        uint3 const r1 = { r0.x + DXTable_d[ direction ],
                           r0.y + DYTable_d[ direction ],
                           r0.z + DZTable_d[ direction ] };

    #ifdef NONPERIODICITY
        if ( T_CoordinateCuda(0) <= r1.x && r1.x < dcBoxXM1 &&
             T_CoordinateCuda(0) <= r1.y && r1.y < dcBoxYM1 &&
             T_CoordinateCuda(0) <= r1.z && r1.z < dcBoxZM1    )
        {
    #endif
            auto const nNeighbors = dpNeighborsSizes[ iOffset + iMonomer ];
            bool forbiddenBond = false;
            for ( auto iNeighbor = decltype( nNeighbors )(0); iNeighbor < nNeighbors; ++iNeighbor )
            {
                auto const iGlobalNeighbor = dpNeighbors[ iNeighbor * rNeighborsPitchElements + iMonomer ];
                auto const data2 = dpPolymerSystem[ iGlobalNeighbor ];
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
    vecIntCUDA    const * const              dpPolymerSystem,
    T_Flags             * const              dpPolymerFlags ,
    T_Lattice           * const __restrict__ dpLattice      ,
    T_Id                  const              nMonomers      ,
    cudaTextureObject_t   const              texLatticeTmp
)
{
    for ( auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x )
    {
        auto const properties = dpPolymerFlags[ iMonomer ];
        if ( ( properties & T_Flags(1) ) == T_Flags(0) )    // impossible move
            continue;

        auto const r0 = dpPolymerSystem[ iMonomer ];
        //uint3 const r0 = { r0Raw.x, r0Raw.y, r0Raw.z }; // slower
        auto const direction = ( properties >> 2 ) & T_Flags(7); // 7=0b111
        uint32_t iOldPos;
    #ifdef USE_BIT_PACKING_TMP_LATTICE
        if ( checkFront( texLatticeTmp, r0.x, r0.y, r0.z, direction, &bitPackedTextureGet< T_Lattice >, &iOldPos ) )
    #else
        if ( checkFront( texLatticeTmp, r0.x, r0.y, r0.z, direction, &tex1Dfetch< T_Lattice >, &iOldPos ) )
    #endif
            continue;

        /* If possible, perform move now on normal lattice */
        dpPolymerFlags[ iMonomer ] = properties | T_Flags(2); // indicating allowed move
        dpLattice[ iOldPos ] = 0;
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
    vecIntCUDA          * const              dpPolymerSystem,
    T_Flags             * const              dpPolymerFlags ,
    T_Lattice           * const __restrict__ dpLattice      ,
    T_Id                  const              nMonomers      ,
    cudaTextureObject_t   const              texLatticeTmp
)
{
    for ( auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x )
    {
        auto const properties = dpPolymerFlags[ iMonomer ];
        if ( ( properties & T_Flags(1) ) == T_Flags(0) )    // impossible move
            continue;

        auto const r0 = dpPolymerSystem[ iMonomer ];
        auto const direction = ( properties >> 2 ) & T_Flags(7); // 7=0b111
        uint32_t iOldPos;
    #ifdef USE_BIT_PACKING_TMP_LATTICE
        if ( checkFront( texLatticeTmp, r0.x, r0.y, r0.z, direction, &bitPackedTextureGet< T_Lattice >, &iOldPos ) )
    #else
        if ( checkFront( texLatticeTmp, r0.x, r0.y, r0.z, direction, &tex1Dfetch< T_Lattice >, &iOldPos ) )
    #endif
            continue;

        dpLattice[ iOldPos ] = 0;
        vecIntCUDA const r1 = {
            T_CoordinateCuda( r0.x + DXTableIntCUDA_d[ direction ] ),
            T_CoordinateCuda( r0.y + DYTableIntCUDA_d[ direction ] ),
            T_CoordinateCuda( r0.z + DZTableIntCUDA_d[ direction ] ),
            T_CoordinateCuda( 0 )
        };
        dpLattice[ linearizeBoxVectorIndex( r1.x, r1.y, r1.z ) ] = 1;
        /* If possible, perform move now on normal lattice */
        dpPolymerSystem[ iMonomer ] = r1;
    }
}

__global__ void kernelCountFilteredPerform
(
    vecIntCUDA       const * const              dpPolymerSystem  ,
    T_Flags          const * const              dpPolymerFlags   ,
    T_Lattice        const * const __restrict__ /* dpLattice */  ,
    T_Id                     const              nMonomers        ,
    cudaTextureObject_t      const              texLatticeTmp    ,
    unsigned long long int * const              dpFiltered
)
{
    for ( auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x )
    {
        auto const properties = dpPolymerFlags[ iMonomer ];
        if ( ( properties & T_Flags(1) ) == T_Flags(0) )    // impossible move
            continue;

        auto const data = dpPolymerSystem[ iMonomer ];
        auto const direction = ( properties >> 2 ) & T_Flags(7); // 7=0b111
        if ( checkFront( texLatticeTmp, data.x, data.y, data.z, direction ) )
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
    vecIntCUDA          * const              dpPolymerSystem,
    T_Flags       const * const              dpPolymerFlags ,
    T_Lattice           * const __restrict__ dpLatticeTmp   ,
    T_Id                  const              nMonomers
)
{
    for ( auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x )
    {
        auto const properties = dpPolymerFlags[ iMonomer ];
        if ( ( properties & T_Flags(3) ) == T_Flags(0) )    // impossible move
            continue;

        auto r0 = dpPolymerSystem[ iMonomer ];
        auto const direction = ( properties >> 2 ) & T_Flags(7); // 7=0b111

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
            dpPolymerSystem[ iMonomer ] = r0;
    }
}


/**
 * find jumps and "deapply" them. We just have to find jumps larger than
 * the number of time steps calculated assuming the monomers can only move
 * 1 cell per time step per direction (meaning this also works with
 * diagonal moves!)
 * If for example the box size is 32, but we also calculate with uint8,
 * then the particles may seemingly move not only bei +-32, but also by
 * +-256, but in both cases the particle actually only moves one virtual
 * box.
 * E.g. the particle was at 0 moved to -1 which was mapped to 255 because
 * uint8 overflowed, but the box size is 64, then deltaMove=255 and we
 * need to subtract 3*64. This only works if the box size is a multiple of
 * the type maximum number (256). I.e. in any sane environment if the box
 * size is a power of two, which was a requirement anyway already.
 * Actually, as the position is just calculated as +-1 without any wrapping,
 * the only way for jumps to happen is because of type overflows.
 */
__global__ void kernelTreatOverflows
(
    T_UCoordinatesCuda * const dpPolymerSystemOld        ,
    T_UCoordinatesCuda * const dpPolymerSystem           ,
    T_Coordinates      * const dpiPolymerSystemVirtualBox,
    T_Id                 const nMonomers
)
{
    for ( auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x )
    {
        auto const r0 = dpPolymerSystemOld        [ iMonomer ];
        auto       r1 = dpPolymerSystem           [ iMonomer ];
        auto       iv = dpiPolymerSystemVirtualBox[ iMonomer ];
        T_Coordinates const dr = {
            r1.x - r0.x,
            r1.y - r0.y,
            r1.z - r0.z
        };

        auto constexpr boxSizeCudaType = 1ll << ( sizeof( T_UCoordinateCuda ) * CHAR_BIT );
        assert( boxSizeCudaType >= dcBoxX );
        assert( boxSizeCudaType >= dcBoxY );
        assert( boxSizeCudaType >= dcBoxZ );
        //assert( nMonteCarloSteps < boxSizeCudaType / 2 );
        //assert( nMonteCarloSteps <= std::min( std::min( mBoxX, mBoxY ), mBoxZ ) / 2 );

        if ( std::abs( dr.x ) > T_UCoordinateCuda( boxSizeCudaType / 2 ) )
        {
            r1.x -= boxSizeCudaType - dcBoxX;
            iv.x -= dr.x > decltype( dr.x )(0) ? 1 : -1;
        }
        if ( std::abs( dr.y ) > T_UCoordinateCuda( boxSizeCudaType / 2 ) )
        {
            r1.y -= boxSizeCudaType - dcBoxY;
            iv.y -= dr.y > decltype( dr.y )(0) ? 1 : -1;
        }
        if ( std::abs( dr.z ) > T_UCoordinateCuda( boxSizeCudaType / 2 ) )
        {
            r1.z -= boxSizeCudaType - dcBoxZ;
            iv.z -= dr.z > decltype( dr.z )(0) ? 1 : -1;
        }

        dpPolymerSystem           [ iMonomer ] = r1;
        dpiPolymerSystemVirtualBox[ iMonomer ] = iv;
    }
}

} // end anonymous namespace with typedefs for kernels



UpdaterGPUScBFM_AB_Type::T_Id UpdaterGPUScBFM_AB_Type::linearizeBoxVectorIndex
(
    T_Coordinate const & ix,
    T_Coordinate const & iy,
    T_Coordinate const & iz
) const
{
    #if defined ( USE_ZCURVE_FOR_LATTICE ) || defined ( USE_MOORE_CURVE_FOR_LATTICE )
        auto const zorder =
              diluteBits< T_Id, 2 >( T_Id( ix ) & mBoxXM1 )        +
            ( diluteBits< T_Id, 2 >( T_Id( iy ) & mBoxYM1 ) << 1 ) +
            ( diluteBits< T_Id, 2 >( T_Id( iz ) & mBoxZM1 ) << 2 );
        #if defined ( USE_MOORE_CURVE_FOR_LATTICE )
            return fromGrayCode( zorder );
        #else
            return zorder;
        #endif
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
        return   ( T_Id( ix ) & mBoxXM1 ) +
               ( ( T_Id( iy ) & mBoxYM1 ) << mBoxXLog2  ) +
               ( ( T_Id( iz ) & mBoxZM1 ) << mBoxXYLog2 );
    #endif
}

UpdaterGPUScBFM_AB_Type::UpdaterGPUScBFM_AB_Type()
 : mStream                          ( 0    ),
   mAge                             ( 0    ),
   mnStepsBetweenSortings           ( 5000 ),
   mLatticeOut                      ( NULL ),
   mLatticeTmp                      ( NULL ),
   mLatticeTmp2                     ( NULL ),
   mnAllMonomers                    ( 0    ),
   mnMonomersPadded                 ( 0    ),
   mPolymerSystem                   ( NULL ),
   mPolymerSystemSorted             ( NULL ),
   mPolymerSystemSortedOld          ( NULL ),
   mviPolymerSystemSortedVirtualBox ( NULL ),
   mPolymerFlags                    ( NULL ),
   miToiNew                         ( NULL ),
   miNewToi                         ( NULL ),
   miNewToiComposition              ( NULL ),
   miNewToiSpatial                  ( NULL ),
   mvKeysZOrderLinearIds            ( NULL ),
   mNeighbors                       ( NULL ),
   mNeighborsSorted                 ( NULL ),
   mNeighborsSortedSizes            ( NULL ),
   mNeighborsSortedInfo             ( nBytesAlignment ),
   mBoxX                            ( 0    ),
   mBoxY                            ( 0    ),
   mBoxZ                            ( 0    ),
   mBoxXM1                          ( 0    ),
   mBoxYM1                          ( 0    ),
   mBoxZM1                          ( 0    ),
   mBoxXLog2                        ( 0    ),
   mBoxXYLog2                       ( 0    )
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


struct DeleteMirroredVector
{
    size_t nBytesFreed = 0;

    template< typename S >
    void operator()( MirroredVector< S > * & p, std::string const & name = "" )
    {
        if ( p != NULL )
        {
            std::cerr
                << "Free MirroredVector " << name << " at " << (void*) p
                << " which holds " << prettyPrintBytes( p->nBytes ) << "\n";
            nBytesFreed += p->nBytes;
            delete p;
            p = NULL;
        }
    }

    template< typename S >
    void operator()( MirroredTexture< S > * & p, std::string const & name = "" )
    {
        if ( p != NULL )
        {
            nBytesFreed += p->nBytes;
            delete p;
            p = NULL;
        }
    }
};

/**
 * Deletes everything which could and is allocated
 */
void UpdaterGPUScBFM_AB_Type::destruct()
{
    DeleteMirroredVector deletePointer;
    deletePointer( mLatticeOut                     , "mLatticeOut"                      );
    deletePointer( mLatticeOut                     , "mLatticeOut"                      );
    deletePointer( mLatticeTmp                     , "mLatticeTmp"                      );
    deletePointer( mLatticeTmp2                    , "mLatticeTmp2"                     );
    deletePointer( mPolymerSystem                  , "mPolymerSystem"                   );
    deletePointer( mPolymerSystemSorted            , "mPolymerSystemSorted"             );
    deletePointer( mPolymerSystemSortedOld         , "mPolymerSystemSortedOld"          );
    deletePointer( mviPolymerSystemSortedVirtualBox, "mviPolymerSystemSortedVirtualBox" );
    deletePointer( mPolymerFlags                   , "mPolymerFlags"                    );
    deletePointer( miToiNew                        , "miToiNew"                         );
    deletePointer( miNewToi                        , "miNewToi"                         );
    deletePointer( miNewToiComposition             , "miNewToiComposition"              );
    deletePointer( miNewToiSpatial                 , "miNewToiSpatial"                  );
    deletePointer( mvKeysZOrderLinearIds           , "mvKeysZOrderLinearIds"            );
    deletePointer( mNeighbors                      , "mNeighbors"                       );
    deletePointer( mNeighborsSorted                , "mNeighborsSorted"                 );
    deletePointer( mNeighborsSortedSizes           , "mNeighborsSortedSizes"            );
    mLog( "Info" ) << "Freed a total of " << prettyPrintBytes( deletePointer.nBytesFreed ) << " on GPU and host RAM.\n";
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

#if 0
{
    /* a class helper for restructuring data */
    template< typename >
    class ShufflingVector
    {
        private:
            size_t const n;
            std::vector< size_t > miToiNew, miNewToi;

        public:
            inline ShufflingVector( size_t const & n )
            :
    }
}
#endif

void UpdaterGPUScBFM_AB_Type::initializeBondTable( void )
{
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
        msg << "[" << __FILENAME__ << "::initializeBondTable] "
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
    uint32_t tmp_DXTable2[6] = { 0u-2u,2,  0,0,  0,0 };
    uint32_t tmp_DYTable2[6] = {  0,0, 0u-2u,2,  0,0 };
    uint32_t tmp_DZTable2[6] = {  0,0,  0,0, 0u-2u,2 };
    CUDA_ERROR( cudaMemcpyToSymbol( DXTable2_d, tmp_DXTable2, sizeof( tmp_DXTable2 ) ) );
    CUDA_ERROR( cudaMemcpyToSymbol( DYTable2_d, tmp_DYTable2, sizeof( tmp_DXTable2 ) ) );
    CUDA_ERROR( cudaMemcpyToSymbol( DZTable2_d, tmp_DZTable2, sizeof( tmp_DXTable2 ) ) );
    T_CoordinateCuda tmp_DXTableIntCUDA[6] = { -1,1,  0,0,  0,0 };
    T_CoordinateCuda tmp_DYTableIntCUDA[6] = {  0,0, -1,1,  0,0 };
    T_CoordinateCuda tmp_DZTableIntCUDA[6] = {  0,0,  0,0, -1,1 };
    CUDA_ERROR( cudaMemcpyToSymbol( DXTableIntCUDA_d, tmp_DXTableIntCUDA, sizeof( tmp_DZTableIntCUDA ) ) );
    CUDA_ERROR( cudaMemcpyToSymbol( DYTableIntCUDA_d, tmp_DYTableIntCUDA, sizeof( tmp_DZTableIntCUDA ) ) );
    CUDA_ERROR( cudaMemcpyToSymbol( DZTableIntCUDA_d, tmp_DZTableIntCUDA, sizeof( tmp_DZTableIntCUDA ) ) );
}

void UpdaterGPUScBFM_AB_Type::initializeSpeciesSorting( void )
{
    mLog( "Info" ) << "Coloring graph ...\n";
    bool const bUniformColors = true; // setting this to true should yield more performance as the kernels are uniformly utilized
    mGroupIds = graphColoring< MonomerEdges const *, T_Id, T_Color >(
        mNeighbors->host, mNeighbors->nElements, bUniformColors,
        []( MonomerEdges const * const & x, T_Id const & i ){ return x[i].size; },
        []( MonomerEdges const * const & x, T_Id const & i, size_t const & j ){ return x[i].neighborIds[j]; }
    );

    /* check automatic coloring with that given in BFM-file */
    if ( mLog.isActive( "Check" ) )
    {
        mLog( "Info" ) << "Checking difference between automatic and given coloring ... ";
        size_t nDifferent = 0;
        for ( size_t iMonomer = 0u; iMonomer < std::max< size_t >( 20, mnAllMonomers ); ++iMonomer )
        {
            if ( int32_t( mGroupIds.at( iMonomer )+1 ) != mAttributeSystem[ iMonomer ] )
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
            mLog( "Stats" ) << char( 'A' + (char) i ) << ": " << mnElementsInGroup[i] << "x (" << (float) mnElementsInGroup[i] / mnAllMonomers * 100.f << "%), ";
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
    mnMonomersPadded = mnAllMonomers + ( nElementsAlignment - 1u ) * mnElementsInGroup.size();

    assert( miToiNew      == NULL );
    assert( miNewToi      == NULL );
    assert( mPolymerFlags == NULL );
    miToiNew      = new MirroredVector< T_Id    >( mnAllMonomers, mStream );
    miNewToi      = new MirroredVector< T_Id    >( mnMonomersPadded, mStream );
    mPolymerFlags = new MirroredVector< T_Flags >( mnMonomersPadded, mStream );
    assert( miToiNew      != NULL );
    assert( miNewToi      != NULL );
    assert( mPolymerFlags != NULL );
    mPolymerFlags->memsetAsync(0); // can do async as it is next needed in runSimulationOnGPU

    /* calculate offsets to each aligned subgroup vector */
    mviSubGroupOffsets.resize( mnElementsInGroup.size() );
    mviSubGroupOffsets.at(0) = 0;
    for ( size_t i = 1u; i < mnElementsInGroup.size(); ++i )
    {
        mviSubGroupOffsets[i] = mviSubGroupOffsets[i-1] +
        ceilDiv( mnElementsInGroup[i-1], nElementsAlignment ) * nElementsAlignment;
        assert( mviSubGroupOffsets[i] - mviSubGroupOffsets[i-1] >= mnElementsInGroup[i-1] );
    }

    /* virtually sort groups into new array and save index mappings */
    auto iSubGroup = mviSubGroupOffsets;   /* stores the next free index for each subgroup */
    for ( size_t i = 0u; i < mnAllMonomers; ++i )
        miToiNew->host[i] = iSubGroup[ mGroupIds[i] ]++;

    /* create convenience reverse mapping */
    std::fill( miNewToi->host, miNewToi->host + miNewToi->nElements, UINT32_MAX );
    for ( size_t iOld = 0u; iOld < mnAllMonomers; ++iOld )
        miNewToi->host[ miToiNew->host[ iOld ] ] = iOld;

    if ( mLog.isActive( "Info" ) )
    {
        mLog( "Info" ) << "mviSubGroupOffsets = { ";
        for ( auto const & x : mviSubGroupOffsets )
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

    #if defined( USE_PERIODIC_MONOMER_SORTING ) || defined( USE_GPU_FOR_OVERHEAD )
        miNewToi->pushAsync();
        miToiNew->pushAsync();
    #endif
}

using MonomerEdges = UpdaterGPUScBFM_AB_Type::MonomerEdges;

/**
 * Calculates mapping BtoA from AtoB mapping
 */
__global__ void kernelInvertMapping
(
    T_Id         const * const dpiNewToi      ,
    T_Id               * const dpiToiNew      ,
    T_Id                 const nMonomersPadded
)
{
    for ( auto iNew = blockIdx.x * blockDim.x + threadIdx.x;
          iNew < nMonomersPadded; iNew += gridDim.x * blockDim.x )
    {
        auto const iOld = dpiNewToi[ iNew ];
        if ( iOld != UINT32_MAX )
            dpiToiNew[ iOld ] = iNew;
    }
}

/**
 * needs to be called for each species
 */
__global__ void kernelApplyMappingToNeighbors
(
    //T_UCoordinatesCuda const * const dpPolymerSystem ,
    MonomerEdges const * const dpNeighbors            ,
    T_Id         const * const dpiNewToi              ,
    T_Id         const * const dpiToiNew              ,
    T_Id               * const dpNeighborsSorted      ,
    uint32_t             const rNeighborsPitchElements,
    uint8_t            * const dpNeighborsSortedSizes ,
    T_Id                 const nMonomers
)
{
    /* apply sorting for polymers, see initializePolymerSystemSorted */

    /* apply sorting for neighbor info, see initializeSortedNeighbors */
    for ( auto iMonomer = blockIdx.x * blockDim.x + threadIdx.x;
          iMonomer < nMonomers; iMonomer += gridDim.x * blockDim.x )
    {
        auto const iOld = dpiNewToi[ iMonomer ];
        auto const nNeighbors = dpNeighbors[ iOld ].size;
        dpNeighborsSortedSizes[ iMonomer ] = nNeighbors;
        for ( size_t j = 0u; j < nNeighbors; ++j )
        {
            dpNeighborsSorted[ j * rNeighborsPitchElements + iMonomer ] =
                dpiToiNew[ dpNeighbors[ iOld ].neighborIds[ j ] ];
        }
    }
}

__global__ void kernelUndoPolymerSystemSorting
(
    T_UCoordinatesCuda const * const dpPolymerSystemSorted           ,
    T_Coordinates      const * const dpiPolymerSystemSortedVirtualBox,
    T_Id               const * const dpiNewToi                       ,
    T_Coordinates            * const dpPolymerSystem                 ,
    T_Id                       const nMonomersPadded
)
{
    for ( auto iNew = blockIdx.x * blockDim.x + threadIdx.x;
          iNew < nMonomersPadded; iNew += gridDim.x * blockDim.x )
    {
        auto const iOld = dpiNewToi[ iNew ];
        if ( iOld == UINT32_MAX )
            continue;
        auto const rsmall = dpPolymerSystemSorted[ iNew ];
        T_Coordinates rSorted = { rsmall.x, rsmall.y, rsmall.z, rsmall.w };
        auto const nPos = dpiPolymerSystemSortedVirtualBox[ iNew ];
        rSorted.x += nPos.x * dcBoxX;
        rSorted.y += nPos.y * dcBoxY;
        rSorted.z += nPos.z * dcBoxZ;
        dpPolymerSystem[ iOld ] = rSorted;
    }
}

__global__ void kernelSplitMonomerPositions
(
    T_Coordinates const * const dpPolymerSystem                 ,
    T_Id          const * const dpiNewToi                       ,
    T_Coordinates       * const dpiPolymerSystemSortedVirtualBox,
    T_UCoordinatesCuda  * const dpPolymerSystemSorted           ,
    size_t                const nMonomersPadded
)
{
    for ( auto iNew = blockIdx.x * blockDim.x + threadIdx.x;
          iNew < nMonomersPadded; iNew += gridDim.x * blockDim.x )
    {
        auto const iOld = dpiNewToi[ iNew ];
        if ( iOld == UINT32_MAX )
            continue;
        auto const r = dpPolymerSystem[ iOld ];
        T_UCoordinatesCuda rlo = {
            T_UCoordinateCuda( r.x & dcBoxXM1 ),
            T_UCoordinateCuda( r.y & dcBoxYM1 ),
            T_UCoordinateCuda( r.z & dcBoxZM1 ),
            T_UCoordinateCuda( dpPolymerSystemSorted[ iNew ].w )
        };
        dpPolymerSystemSorted[ iNew ] = rlo;
        T_Coordinates rhi = {
            ( r.x - T_Coordinate( rlo.x ) ) / T_Coordinate( dcBoxX ),
            ( r.y - T_Coordinate( rlo.y ) ) / T_Coordinate( dcBoxY ),
            ( r.z - T_Coordinate( rlo.z ) ) / T_Coordinate( dcBoxZ ),
            0
        };
        dpiPolymerSystemSortedVirtualBox[ iNew ] = rhi;
    }
}


struct LinearizeBoxVectorIndexFunctor
{
    __device__ inline T_Id operator()( T_UCoordinatesCuda const & r ) const
    {
        return linearizeBoxVectorIndex( r.x, r.y, r.z );
    }
};

/**
 * this works on mPolymerSystemSorted and resorts the monomers along a
 * z-order curve in order to improve cache hit rates, especially for "slow"
 * systems. Also it updates the order of and the IDs inside mNeighborsSorted
 * @param[in]  polymerSystemSorted
 * @param[in]  iToiNew specifies the mapping used to create polymerSystemSorted
 *             from polymerSystem, i.e.
 *             polymerSystemSorted[ iToiNew[i] ] == polymerSystem[i]
 * @param[in]  iNewToi same as iToiNew, but the other way arount, i.e.
 *             polymerSystemSorted[i] == polymerSystem[ iNewToi ]
 *             Note that the sorted system includes padding, therefore some
 *             entries of iNewToi contain UINT32_MAX to indicate that those
 *             are not do be mapped
 * @param[out] iToiNew just as the input, but after sorting spatially
 * @param[out] iNewToi
 */
void UpdaterGPUScBFM_AB_Type::doSpatialSorting( void )
{
    auto const nThreads = 128;
    auto const nBlocksP = ceilDiv( mnMonomersPadded, nThreads );
    /* because resorting changes the order we have to do the full
     * overflow checks and also update mPolymerSystemSortedOld ! */
    #if defined( USE_UINT8_POSITIONS )
    {
        /* the padding values do not change, so we can simply let the threads
         * calculate them without worries and save the loop over the species */
        kernelTreatOverflows<<< nBlocksP, nThreads, 0, mStream >>>(
            mPolymerSystemSortedOld         ->gpu,
            mPolymerSystemSorted            ->gpu,
            mviPolymerSystemSortedVirtualBox->gpu,
            mnMonomersPadded
        );
    }
    #endif

    /* dependent on kernelTreatOverflows */
    kernelUndoPolymerSystemSorting<<< nBlocksP, nThreads, 0, mStream >>>
    (
        mPolymerSystemSorted            ->gpu,
        mviPolymerSystemSortedVirtualBox->gpu,
        miNewToi                        ->gpu,
        mPolymerSystem                  ->gpu,
        mnMonomersPadded
    );

    /* mapping new (monomers spatially sorted) index to old (species sorted) index */
    if ( miNewToiComposition   == NULL ) miNewToiComposition   = new MirroredVector< T_Id >( mnMonomersPadded, mStream );
    if ( miNewToiSpatial       == NULL ) miNewToiSpatial       = new MirroredVector< T_Id >( mnMonomersPadded, mStream );
    if ( mvKeysZOrderLinearIds == NULL ) mvKeysZOrderLinearIds = new MirroredVector< T_Id >( mnMonomersPadded, mStream );
    assert( miNewToiComposition   != NULL );
    assert( miNewToiSpatial       != NULL );
    assert( mvKeysZOrderLinearIds != NULL );

    /* @see https://thrust.github.io/doc/group__transformations.html#ga233a3db0c5031023c8e9385acd4b9759
       @see https://thrust.github.io/doc/group__transformations.html#ga281b2e453bfa53807eda1d71614fb504 */
    /* not dependent on anything, could run in different stream */
    thrust::sequence( thrust::system::cuda::par, miNewToiSpatial->gpu, miNewToiSpatial->gpu + miNewToiSpatial->nElements );
    /* dependent on above, but does not depend on kernelUndoPolymerSystemSorting */
    thrust::transform( thrust::system::cuda::par,
        mPolymerSystemSorted ->gpu,
        mPolymerSystemSorted ->gpu + mPolymerSystemSorted->nElements,
        mvKeysZOrderLinearIds->gpu,
        LinearizeBoxVectorIndexFunctor()
    );
    /* sort per sublists (each species) by key, not the whole list */
    for ( auto iSpecies = 0u; iSpecies < mnElementsInGroup.size(); ++iSpecies )
    {
        thrust::sort_by_key( thrust::system::cuda::par,
            mvKeysZOrderLinearIds->gpu + mviSubGroupOffsets.at( iSpecies ),
            mvKeysZOrderLinearIds->gpu + mviSubGroupOffsets.at( iSpecies ) + mnElementsInGroup.at( iSpecies ),
            miNewToiSpatial      ->gpu + mviSubGroupOffsets.at( iSpecies )
        );
    }

    thrust::fill( thrust::system::cuda::par, miNewToiComposition->gpu, miNewToiComposition->gpu + miNewToiComposition->nElements, UINT32_MAX );
    /**
     * @see https://thrust.github.io/doc/group__gathering.html#ga86722e76264fb600d659c1adef5d51b2
     *   -> for ( it : map ) result[ it - map_first ] = input_first[ *it ]
     *   -> for ( i ) result[i] = input_first[ map[i] ]
     * for ( T_Id iNew = 0u; iNew < miNewToiSpatial->nElements ; ++iNew )
     *     iNewToiComposition.at( iNew ) = miNewToi->host[ miNewToiSpatial->host[ iNew ] ];
     */
    thrust::gather( thrust::system::cuda::par,
        miNewToiSpatial   ->gpu,
        miNewToiSpatial   ->gpu + miNewToiSpatial->nElements,
        miNewToi          ->gpu,
        miNewToiComposition->gpu
    );
    std::swap( miNewToi->gpu, miNewToiComposition->gpu ); // avoiding memcpy by swapping pointers on GPU
    kernelInvertMapping<<< nBlocksP, nThreads >>>( miNewToi->gpu, miToiNew->gpu, miNewToi->nElements );
    for ( auto iSpecies = 0u; iSpecies < mnElementsInGroup.size(); ++iSpecies )
    {
        kernelApplyMappingToNeighbors<<< nBlocksP, nThreads, 0, mStream >>>(
            mNeighbors           ->gpu,
            miNewToi             ->gpu + mviSubGroupOffsets[ iSpecies ],
            miToiNew             ->gpu,
            mNeighborsSorted     ->gpu + mNeighborsSortedInfo.getMatrixOffsetElements( iSpecies ),
            mNeighborsSortedInfo.getMatrixPitchElements( iSpecies ),
            mNeighborsSortedSizes->gpu + mviSubGroupOffsets[ iSpecies ],
            mnElementsInGroup[ iSpecies ]
        );
    }

    /* kernelUndoPolymerSystemSorting followed by kernelSplitMonomerPositions
     * basically just avoids using two temporary arrays for the resorting of
     * mPolymerSystemSorted and mviPolymerSystemSortedVirtualBox */

    /* dependent on:
     *   kernelUndoPolymerSystemSorting (mPolymerSystem)
     *   thrust::transform (mPolymerSystemSorted)
     *   thrust::gather (miNewToi)
     */
    kernelSplitMonomerPositions<<< nBlocksP, nThreads, 0, mStream >>>(
        mPolymerSystem                  ->gpu,
        miNewToi                        ->gpu,
        mviPolymerSystemSortedVirtualBox->gpu,
        mPolymerSystemSorted            ->gpu,
        mnMonomersPadded
    );

    CUDA_ERROR( cudaMemcpyAsync( mPolymerSystemSortedOld->gpu, mPolymerSystemSorted->gpu, mPolymerSystemSortedOld->nBytes, cudaMemcpyDeviceToDevice, mStream ) );
}

void UpdaterGPUScBFM_AB_Type::initializeSortedNeighbors( void )
{
    /* adjust neighbor IDs to new sorted PolymerSystem and also sort that array.
     * Bonds are not supposed to change, therefore we don't need to push and
     * pop them each time we do something on the GPU! */

    assert( mNeighborsSortedInfo.getRequiredBytes() == 0 );
    for ( size_t i = 0u; i < mnElementsInGroup.size(); ++i )
        mNeighborsSortedInfo.newMatrix( MAX_CONNECTIVITY, mnElementsInGroup[i] );
    if ( mNeighborsSorted      == NULL ) mNeighborsSorted      = new MirroredVector< T_Id    >( mNeighborsSortedInfo.getRequiredElements(), mStream );
    if ( mNeighborsSortedSizes == NULL ) mNeighborsSortedSizes = new MirroredVector< uint8_t >( mnMonomersPadded, mStream );
    assert( mNeighborsSorted      != NULL );
    assert( mNeighborsSortedSizes != NULL );

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
        mLog( "Info" ) << "[UpdaterGPUScBFM_AB_Type::initializeSortedNeighbors] map neighborIds to sorted array ... ";
    }


#if defined( USE_GPU_FOR_OVERHEAD )
    auto const nThreads = 128;
    auto const nBlocksP = ceilDiv( mnMonomersPadded, nThreads );
    for ( auto iSpecies = 0u; iSpecies < mnElementsInGroup.size(); ++iSpecies )
    {
        kernelApplyMappingToNeighbors<<< nBlocksP, nThreads, 0, mStream >>>(
            mNeighbors           ->gpu,
            miNewToi             ->gpu + mviSubGroupOffsets[ iSpecies ],
            miToiNew             ->gpu,
            mNeighborsSorted     ->gpu + mNeighborsSortedInfo.getMatrixOffsetElements( iSpecies ),
            mNeighborsSortedInfo.getMatrixPitchElements( iSpecies ),
            mNeighborsSortedSizes->gpu + mviSubGroupOffsets[ iSpecies ],
            mnElementsInGroup[ iSpecies ]
        );
    }
#else
    for ( auto iSpecies = 0u; iSpecies < mnElementsInGroup.size(); ++iSpecies )
    {
        for ( size_t iMonomer = 0u; iMonomer < mnElementsInGroup[ iSpecies ]; ++iMonomer )
        {
            auto const i = mviSubGroupOffsets[ iSpecies ] + iMonomer;
            auto const iOld = miNewToi->host[i];

            mNeighborsSortedSizes->host[i] = mNeighbors->host[ iOld ].size;
            auto const pitch = mNeighborsSortedInfo.getMatrixPitchElements( iSpecies );
            for ( size_t j = 0u; j < mNeighbors->host[ iOld ].size; ++j )
            {
                if ( i < 5 || std::abs( (long long int) i - mviSubGroupOffsets[ mviSubGroupOffsets.size()-1 ] ) < 5 )
                {
                    mLog( "Info" ) << "Currently at index " << i << ": Writing into mNeighborsSorted->host[ " << mNeighborsSortedInfo.getMatrixOffsetElements( iSpecies ) << " + " << j << " * " << pitch << " + " << i << "-" << mviSubGroupOffsets[ iSpecies ] << "] the value of old neighbor located at miToiNew->host[ mNeighbors[ miNewToi->host[i]=" << miNewToi->host[i] << " ] = miToiNew->host[ " << mNeighbors->host[ miNewToi->host[i] ].neighborIds[j] << " ] = " << miToiNew->host[ mNeighbors->host[ miNewToi->host[i] ].neighborIds[j] ] << " \n";
                }
                auto const iNeighborSorted = mNeighborsSortedInfo.getMatrixOffsetElements( iSpecies )
                                           + j * pitch + iMonomer;
                mNeighborsSorted->host[ iNeighborSorted ] = miToiNew->host[ mNeighbors->host[ iOld ].neighborIds[ j ] ];
            }
        }
    }

    mNeighborsSorted     ->pushAsync();
    mNeighborsSortedSizes->pushAsync();
    mLog( "Info" ) << "Done\n";
#endif

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
                if ( iSorted >= mnMonomersPadded )
                    throw std::runtime_error( "New index out of range!" );
            }
        }*/
        /* does a similar check for the unsorted error which is still used
         * to create the property tag */
        for ( T_Id i = 0; i < mnAllMonomers; ++i )
        {
            if ( mNeighbors->host[i].size > MAX_CONNECTIVITY )
            {
                std::stringstream msg;
                msg << "[" << __FILENAME__ << "::initializeSortedNeighbors] "
                    << "This implementation allows max. 7 neighbors per monomer, "
                    << "but monomer " << i << " has " << mNeighbors->host[i].size << "\n";
                mLog( "Error" ) << msg.str();
                throw std::invalid_argument( msg.str() );
            }
        }
    }
}


void UpdaterGPUScBFM_AB_Type::initializeSortedMonomerPositions( void )
{
    /* sort groups into new array and save index mappings */
    if ( mPolymerSystemSorted             == NULL )mPolymerSystemSorted             = new MirroredVector< T_UCoordinatesCuda >( mnMonomersPadded, mStream );
    if ( mPolymerSystemSortedOld          == NULL )mPolymerSystemSortedOld          = new MirroredVector< T_UCoordinatesCuda >( mnMonomersPadded, mStream );
    if ( mviPolymerSystemSortedVirtualBox == NULL )mviPolymerSystemSortedVirtualBox = new MirroredVector< T_Coordinates      >( mnMonomersPadded, mStream );
    assert( mPolymerSystemSorted             != NULL );
    assert( mPolymerSystemSortedOld          != NULL );
    assert( mviPolymerSystemSortedVirtualBox != NULL );
    #ifndef NDEBUG
        mPolymerSystemSorted            ->memset( 0 );
        mPolymerSystemSortedOld         ->memset( 0 );
        mviPolymerSystemSortedVirtualBox->memset( 0 );
    #endif

#if defined( USE_GPU_FOR_OVERHEAD )
    auto const nThreads = 128;
    auto const nBlocksP = ceilDiv( mnMonomersPadded, nThreads );
    kernelSplitMonomerPositions<<< nBlocksP, nThreads, 0, mStream >>>(
        mPolymerSystem                  ->gpu,
        miNewToi                        ->gpu,
        mviPolymerSystemSortedVirtualBox->gpu,
        mPolymerSystemSorted            ->gpu,
        mnMonomersPadded
    );
#else
    mLog( "Info" ) << "[" << __FILENAME__ << "::initializeSortedMonomerPositions] sort mPolymerSystem -> mPolymerSystemSorted ...\n";
    for ( T_Id i = 0u; i < mnAllMonomers; ++i )
    {
        if ( i < 20 )
            mLog( "Info" ) << "Write " << i << " to " << this->miToiNew->host[i] << "\n";

        auto const x = mPolymerSystem->host[i].x;
        auto const y = mPolymerSystem->host[i].y;
        auto const z = mPolymerSystem->host[i].z;

        mPolymerSystemSorted->host[ miToiNew->host[i] ].x = x & mBoxXM1;
        mPolymerSystemSorted->host[ miToiNew->host[i] ].y = y & mBoxYM1;
        mPolymerSystemSorted->host[ miToiNew->host[i] ].z = z & mBoxZM1;
        mPolymerSystemSorted->host[ miToiNew->host[i] ].w = mNeighbors->host[i].size;

        mviPolymerSystemSortedVirtualBox->host[ miToiNew->host[i] ].x = ( x - ( x & mBoxXM1 ) ) / mBoxX;
        mviPolymerSystemSortedVirtualBox->host[ miToiNew->host[i] ].y = ( y - ( y & mBoxYM1 ) ) / mBoxY;
        mviPolymerSystemSortedVirtualBox->host[ miToiNew->host[i] ].z = ( z - ( z & mBoxZM1 ) ) / mBoxZ;

        auto const pTarget  = &mPolymerSystemSorted            ->host[ miToiNew->host[i] ];
        auto const pTarget2 = &mviPolymerSystemSortedVirtualBox->host[ miToiNew->host[i] ];
        if ( ! ( ( (T_Coordinate) pTarget->x + (T_Coordinate) pTarget2->x * (T_Coordinate) mBoxX == x ) &&
                 ( (T_Coordinate) pTarget->y + (T_Coordinate) pTarget2->y * (T_Coordinate) mBoxY == y ) &&
                 ( (T_Coordinate) pTarget->z + (T_Coordinate) pTarget2->z * (T_Coordinate) mBoxZ == z )
        ) )
        {
            std::stringstream msg;
            msg << "[" << __FILENAME__ << "::initializeSortedMonomerPositions] "
                << "Error while trying to compress the globale positions into box size modulo and number of virtual box the monomer resides in. Virtual box number "
                << "(" << pTarget2->x << "," << pTarget2->y << "," << pTarget2->z << ")"
                << ", wrapped position: "
                << "(" << pTarget->x << "," << pTarget->y << "," << pTarget->z << ")"
                << " => reconstructed global position ("
                << pTarget->x + pTarget2->x * mBoxX << ","
                << pTarget->y + pTarget2->y * mBoxY << ","
                << pTarget->z + pTarget2->z * mBoxZ << ")"
                << " should be equal to the input position: "
                << "(" << x << "," << y << "," << z << ")"
                << std::endl;
            throw std::runtime_error( msg.str() );
        }
    }
    mPolymerSystemSorted            ->pushAsync();
    mviPolymerSystemSortedVirtualBox->pushAsync();
#endif
}

void UpdaterGPUScBFM_AB_Type::initializeLattices( void )
{
    if ( mLatticeOut != NULL || mLatticeTmp != NULL )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::initializeLattices] "
            << "Initialize was already called and may not be called again "
            << "until cleanup was called!";
        mLog( "Error" ) << msg.str();
        throw std::runtime_error( msg.str() );
    }

    size_t nBytesLatticeTmp = mBoxX * mBoxY * mBoxZ / sizeof( T_Lattice );
    #if defined( USE_BIT_PACKING_TMP_LATTICE )
        nBytesLatticeTmp /= CHAR_BIT;
    #endif
    #if defined( USE_NBUFFERED_TMP_LATTICE )
        nBytesLatticeTmp *= mnLatticeTmpBuffers;
    #endif
    mLatticeOut  = new MirroredTexture< T_Lattice >( mBoxX * mBoxY * mBoxZ, mStream );
    mLatticeTmp  = new MirroredTexture< T_Lattice >( nBytesLatticeTmp     , mStream );
    mLatticeTmp2 = new MirroredTexture< T_Lattice >( mBoxX * mBoxY * mBoxZ, mStream );
    mLatticeTmp ->memsetAsync(0); // async as it is next needed in runSimulationOnGPU
    mLatticeTmp2->memsetAsync(0);
    /* populate latticeOut with monomers from mPolymerSystem */
    std::memset( mLatticeOut->host, 0, mLatticeOut->nBytes );
    for ( T_Id iMonomer = 0; iMonomer < mnAllMonomers; ++iMonomer )
    {
        mLatticeOut->host[ linearizeBoxVectorIndex(
            mPolymerSystem->host[ iMonomer ].x,
            mPolymerSystem->host[ iMonomer ].y,
            mPolymerSystem->host[ iMonomer ].z
        ) ] = 1;
    }
    mLatticeOut->pushAsync();

    mLog( "Info" )
        << "Filling Rate: " << mnAllMonomers << " "
        << "(=" << mnAllMonomers / 1024 << "*1024+" << mnAllMonomers % 1024 << ") "
        << "particles in a (" << mBoxX << "," << mBoxY << "," << mBoxZ << ") box "
        << "=> " << 100. * mnAllMonomers / ( mBoxX * mBoxY * mBoxZ ) << "%\n"
        << "Note: densest packing is: 25% -> in this case it might be more reasonable to actually iterate over the spaces where particles can move to, keeping track of them instead of iterating over the particles\n";

    #if defined( USE_NBUFFERED_TMP_LATTICE )
        /**
         * Addresses must be aligned to 32=2*4*4 byte boundaries
         * @see https://devtalk.nvidia.com/default/topic/975906/cuda-runtime-api-error-74-misaligned-address/?offset=5
         * Currently the code does not bother with padding the tmp lattice
         * buffers assuming that the box is large enough to automatically
         * lead to the correct alignment. This also assumes the box size to be
         * of power 2
         */
        if ( mBoxX * mBoxY * mBoxZ < 32 )
        {
            std::stringstream msg;
            msg << "[" << __FILENAME__ << "::initializeLattices] [Error] The total cells in the box " << mBoxX * mBoxY * mBoxZ << " is smaller than 32. This is not allowed (yet) with USE_NBUFFERED_TMP_LATTICE turned as it would neccessitate additional padding between the buffers. Please undefine USE_NBUFFERED_TMP_LATTICE in the source code or increase the box size!\n";
            mLog( "Error" ) << msg.str();
            throw std::runtime_error( msg.str() );
        }
        /**
         * "CUDA C Programming Guide 5.0", p73 says "Any address of a variable residing in global memory or returned by one of the memory allocation routines from the driver or runtime API is always aligned to at least 256 bytes"
         * @see https://stackoverflow.com/questions/14082964/cuda-alignment-256bytes-seriously
         */
        else if ( mBoxX * mBoxY * mBoxZ < 256 )
        {
            std::stringstream msg;
            msg << "[" << __FILENAME__ << "::initializeLattices] [Warning] The total cells in the box " << mBoxX * mBoxY * mBoxZ << " is smaller than 256. This might lead to performance loss. Try undefining USE_NBUFFERED_TMP_LATTICE in the source code or increase the box size.\n";
            mLog( "Warning" ) << msg.str();
        }

        mvtLatticeTmp.resize( mnLatticeTmpBuffers );
        cudaResourceDesc mResDesc;
        cudaTextureDesc  mTexDesc;
        std::memset( &mResDesc, 0, sizeof( mResDesc ) );
        mResDesc.resType = cudaResourceTypeLinear;
        mResDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
        mResDesc.res.linear.desc.x = sizeof( mLatticeTmp->gpu[0] ) * CHAR_BIT; // bits per channel
        std::memset( &mTexDesc, 0, sizeof( mTexDesc ) );
        mTexDesc.readMode = cudaReadModeElementType;
        for ( auto i = 0u; i < mnLatticeTmpBuffers; ++i )
        {
            mResDesc.res.linear.sizeInBytes = mBoxX * mBoxY * mBoxZ * sizeof( mLatticeTmp->gpu[0] );
            #if defined( USE_BIT_PACKING_TMP_LATTICE )
                mResDesc.res.linear.sizeInBytes /= CHAR_BIT;
            #endif
            mResDesc.res.linear.devPtr = (uint8_t*) mLatticeTmp->gpu + i * mResDesc.res.linear.sizeInBytes;
            mLog( "Info" )
                << "Bind texture for " << i << "-th temporary lattice buffer "
                << "to mLatticeTmp->gpu + " << ( i * mResDesc.res.linear.sizeInBytes )
                << "\n";
            /* the last three arguments are pointers to constants! */
            cudaCreateTextureObject( &mvtLatticeTmp.at(i), &mResDesc, &mTexDesc, NULL );
        }
    #endif
}

void UpdaterGPUScBFM_AB_Type::checkMonomerReorderMapping( void )
{
    if ( miToiNew->nElements != mnAllMonomers )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::checkMonomerReorderMapping] "
            << "miToiNew must have " << mnAllMonomers << " elements "
            << "(as many as monomers), but it has " << miToiNew->nElements << "!";
        mLog( "Error" ) << msg.str();
        throw std::runtime_error( msg.str() );
    }

    if ( miNewToi->nElements != mnMonomersPadded )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::checkMonomerReorderMapping] "
            << "miNewToi must have " << mnMonomersPadded << " elements "
            << "(as many as monomers), but it has " << miNewToi->nElements << "!";
        mLog( "Error" ) << msg.str();
        throw std::runtime_error( msg.str() );
    }

    auto const nMonomers       = miToiNew->nElements;
    auto const nMonomersPadded = miNewToi->nElements;

    /* check that the mapping is bijective if we exclude
     * entries equal to UINT32_MAX */
    std::vector< bool > vIsMapped( nMonomers, false );

    for ( size_t iNew = 0u; iNew < miNewToi->nElements; ++iNew )
    {
        auto const iOld = miNewToi->host[ iNew ];

        if ( iOld == UINT32_MAX )
            continue;

        if ( ! ( /* 0 <= iOld && */ iOld < nMonomers ) )
        {
            std::stringstream msg;
            msg << "[" << __FILENAME__ << "::checkMonomerReorderMapping] "
                << "New index " << iNew << " is mapped back to " << iOld
                << ", which is out of range [0," << nMonomers-1 << "]";
            mLog( "Error" ) << msg.str();
            throw std::runtime_error( msg.str() );
        }

        if ( vIsMapped.at( iOld ) )
        {
            std::stringstream msg;
            msg << "[" << __FILENAME__ << "::checkMonomerReorderMapping] "
                << "When trying to test whether we can copy back the monomer "
                << "at the sorted index " << iNew << ", it was found that the "
                << "index " << iOld << " in the unsorted array was already "
                << "written to, i.e., we would loose information on copy-back!";
            mLog( "Error" ) << msg.str();
            throw std::runtime_error( msg.str() );
        }
        vIsMapped.at( iOld ) = true;
    }

    size_t const nMapped = std::count( vIsMapped.begin(), vIsMapped.end(), true );
    if ( nMapped != nMonomers )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::checkMonomerReorderMapping] "
            << "The mapping created e.g. by the sorting by species is missing "
            << "some monomers! Only " << nMapped << " / " << nMonomers
            << " are actually mapped to the new sorted array!";
        mLog( "Error" ) << msg.str();
        throw std::runtime_error( msg.str() );
    }

    std::vector< bool > vIsMappedTo( nMonomersPadded, false );

    for ( size_t iOld = 0u; iOld < miToiNew->nElements; ++iOld )
    {
        auto const iNew = miToiNew->host[ iOld ];

        if ( iNew == UINT32_MAX )
            continue;

        if ( ! ( /* 0 <= iNew && */ iNew < mnMonomersPadded ) )
        {
            std::stringstream msg;
            msg << "[" << __FILENAME__ << "::checkMonomerReorderMapping] "
                << "Old index " << iOld << " maps to " << iNew << ", which is "
                << "out of range [0," << mnMonomersPadded-1 << "]";
            mLog( "Error" ) << msg.str();
            throw std::runtime_error( msg.str() );
        }

        if ( vIsMappedTo.at( iNew ) )
        {
            std::stringstream msg;
            msg << "[" << __FILENAME__ << "::checkMonomerReorderMapping] "
                << "When trying to test whether we can copy back the monomer "
                << "at the sorted index " << iNew << ", it was found that the "
                << "index " << iOld << " in the unsorted array was already "
                << "written to, i.e., we would loose information on copy-back!";
            mLog( "Error" ) << msg.str();
            throw std::runtime_error( msg.str() );
        }
        vIsMappedTo.at( iNew ) = true;
    }

    size_t const nMappedTo = std::count( vIsMappedTo.begin(), vIsMappedTo.end(), true );
    if ( nMappedTo != nMonomers )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::checkMonomerReorderMapping] "
            << "The mapping created e.g. by the sorting by species is missing "
            << "some monomers! The sorted array only has " << nMappedTo << " / "
            << nMonomers << " monomers!";
        mLog( "Error" ) << msg.str();
        throw std::runtime_error( msg.str() );
    }

    /* check that it actually is the inverse */
    for ( size_t iOld = 0u; iOld < miToiNew->nElements; ++iOld )
    {
        if ( miToiNew->host[ iOld ] == UINT32_MAX )
            continue;

        if ( miNewToi->host[ miToiNew->host[ iOld ] ] != iOld )
        {
            std::stringstream msg;
            msg << "[" << __FILENAME__ << "::checkMonomerReorderMapping] "
                << "Roundtrip iOld -> iNew -> iOld not working for iOld= "
                << iOld << " -> " << miToiNew->host[ iOld ] << " -> "
                << miNewToi->host[ miToiNew->host[ iOld ] ];
            mLog( "Error" ) << msg.str();
            throw std::runtime_error( msg.str() );
        }
    }

    /* check that it actually is the inverse the other way around */
    for ( size_t iNew = 0u; iNew < miNewToi->nElements; ++iNew )
    {
        if ( miNewToi->host[ iNew ] == UINT32_MAX )
            continue;

        if ( miToiNew->host[ miNewToi->host[ iNew ] ] != iNew )
        {
            std::stringstream msg;
            msg << "[" << __FILENAME__ << "::checkMonomerReorderMapping] "
                << "Roundtrip iNew -> iOld -> iNew not working for iNew= "
                << iNew << " -> " << miNewToi->host[ iNew ] << " -> "
                << miToiNew->host[ miNewToi->host[ iNew ] ];
            mLog( "Error" ) << msg.str();
            throw std::runtime_error( msg.str() );
        }
    }
}

void UpdaterGPUScBFM_AB_Type::initialize( void )
{
    if ( mLog( "Stats" ).isActive() )
    {
        // this is called in parallel it seems, therefore need to buffer it
        std::stringstream msg; msg
        << "[" << __FILENAME__ << "::initialize] The "
        << "(" << mBoxX << "," << mBoxY << "," << mBoxZ << ")"
        << " lattice is populated by " << mnAllMonomers
        << " resulting in a filling rate of "
        << mnAllMonomers / double( mBoxX * mBoxY * mBoxZ ) << "\n";
        mLog( "Stats" ) << msg.str();
    }

    mLog( "Info" )
    << "T_BoxSize          = " << getTypeInfoString< T_BoxSize          >() << "\n"
    << "T_Coordinate       = " << getTypeInfoString< T_Coordinate       >() << "\n"
    << "T_CoordinateCuda   = " << getTypeInfoString< T_CoordinateCuda   >() << "\n"
    << "T_UCoordinateCuda  = " << getTypeInfoString< T_UCoordinateCuda  >() << "\n"
    << "T_Coordinates      = " << getTypeInfoString< T_Coordinates      >() << "\n"
    << "T_CoordinatesCuda  = " << getTypeInfoString< T_CoordinatesCuda  >() << "\n"
    << "T_UCoordinatesCuda = " << getTypeInfoString< T_UCoordinatesCuda >() << "\n"
    << "T_Color            = " << getTypeInfoString< T_Color            >() << "\n"
    << "T_Flags            = " << getTypeInfoString< T_Flags            >() << "\n"
    << "T_Id               = " << getTypeInfoString< T_Id               >() << "\n"
    << "T_Lattice          = " << getTypeInfoString< T_Lattice          >() << "\n";

    auto constexpr maxBoxSize = ( 1llu << ( CHAR_BIT * sizeof( T_CoordinateCuda ) ) );
    if ( mBoxX > maxBoxSize || mBoxY > maxBoxSize || mBoxZ > maxBoxSize )
    {
        std::stringstream msg;
        msg << "The box size is limited to " << maxBoxSize << " in each direction"
            << ", because of the chosen type for T_Coordinate = "
            << getTypeInfoString< T_Coordinate >() << ", but the chose box size is: ("
            << mBoxX << "," << mBoxY << "," << mBoxZ << ")!\n"
            << "Please change T_Coordinate to a larger type if you want to simulate this setup.";
        throw std::runtime_error( msg.str() );
    }

    /**
     * "When you execute asynchronous CUDA commands without specifying
     * a stream, * the runtime uses the default stream. Before CUDA 7,
     * the default stream is  * a special stream which implicitly
     * synchronizes with all other streams on the device."
     * @see https://devblogs.nvidia.com/gpu-pro-tip-cuda-7-streams-simplify-concurrency/
     */
    if ( mStream == 0 )
        CUDA_ERROR( cudaStreamCreate( &mStream ) );

    { decltype( dcBoxX      ) x = mBoxX     ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxX     , &x, sizeof(x) ) ); }
    { decltype( dcBoxY      ) x = mBoxY     ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxY     , &x, sizeof(x) ) ); }
    { decltype( dcBoxZ      ) x = mBoxZ     ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxZ     , &x, sizeof(x) ) ); }
    { decltype( dcBoxXM1    ) x = mBoxXM1   ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXM1   , &x, sizeof(x) ) ); }
    { decltype( dcBoxYM1    ) x = mBoxYM1   ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxYM1   , &x, sizeof(x) ) ); }
    { decltype( dcBoxZM1    ) x = mBoxZM1   ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxZM1   , &x, sizeof(x) ) ); }
    { decltype( dcBoxXLog2  ) x = mBoxXLog2 ; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXLog2 , &x, sizeof(x) ) ); }
    { decltype( dcBoxXYLog2 ) x = mBoxXYLog2; CUDA_ERROR( cudaMemcpyToSymbol( dcBoxXYLog2, &x, sizeof(x) ) ); }

    #if defined( USE_GPU_FOR_OVERHEAD )
        mPolymerSystem->pushAsync();
        mNeighbors    ->pushAsync();
    #endif

    initializeBondTable();
    initializeSpeciesSorting(); /* using miNewToi and miToiNew the monomers are mapped to be sorted by species */
    checkMonomerReorderMapping();
    initializeSortedNeighbors();
    initializeSortedMonomerPositions();
    checkSystem();
    initializeLattices();
    if ( mAge != 0 )
        doSpatialSorting();

    CUDA_ERROR( cudaGetDevice( &miGpuToUse ) );
    CUDA_ERROR( cudaGetDeviceProperties( &mCudaProps, miGpuToUse ) );
}


void UpdaterGPUScBFM_AB_Type::copyBondSet
( int dx, int dy, int dz, bool bondForbidden )
{
    mForbiddenBonds[ linearizeBondVectorIndex(dx,dy,dz) ] = bondForbidden;
}

void UpdaterGPUScBFM_AB_Type::setNrOfAllMonomers( T_Id const rnAllMonomers )
{
    if ( mnAllMonomers != 0 )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setNrOfAllMonomers] "
            << "Number of Monomers already set to " << mnAllMonomers << "!\n";
        throw std::runtime_error( msg.str() );
    }

    mnAllMonomers = rnAllMonomers;
    mAttributeSystem.resize( mnAllMonomers   );
    mNeighbors     = new MirroredVector< MonomerEdges  >( mnAllMonomers );
    mPolymerSystem = new MirroredVector< T_Coordinates >( mnAllMonomers );
    std::memset( mNeighbors->host, 0, mNeighbors->nBytes );
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

void UpdaterGPUScBFM_AB_Type::setAttribute( T_Id i, int32_t attribute ){ mAttributeSystem.at(i) = attribute; }

void UpdaterGPUScBFM_AB_Type::setMonomerCoordinates
(
    T_Id          const i,
    T_Coordinate  const x,
    T_Coordinate  const y,
    T_Coordinate  const z
)
{
#if DEBUG_UPDATERGPUSCBFM_AB_TYPE > 1
    //if ( ! ( 0 <= x && (T_BoxSize) x < mBoxX ) )
    //    std::cout << i << ": (" << x << "," << y << "," << z << ")\n";
    /* can I apply periodic modularity here to allow the full range ??? */
    if ( ! inRange< decltype( mPolymerSystem->host[0].x ) >(x) ||
         ! inRange< decltype( mPolymerSystem->host[0].y ) >(y) ||
         ! inRange< decltype( mPolymerSystem->host[0].z ) >(z)    )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setMonomerCoordinates" << "] "
            << "One or more of the given coordinates "
            << "(" << x << "," << y << "," << z << ") "
            << "is larger than the internal integer data type for "
            << "representing positions allow! (" << std::numeric_limits< T_Coordinate >::min()
            << " <= size <= " << std::numeric_limits< T_Coordinate >::max() << ")";
        throw std::invalid_argument( msg.str() );
    }
#endif
    mPolymerSystem->host[i].x = x;
    mPolymerSystem->host[i].y = y;
    mPolymerSystem->host[i].z = z;
}

int32_t UpdaterGPUScBFM_AB_Type::getMonomerPositionInX( T_Id i ){ return mPolymerSystem->host[i].x; }
int32_t UpdaterGPUScBFM_AB_Type::getMonomerPositionInY( T_Id i ){ return mPolymerSystem->host[i].y; }
int32_t UpdaterGPUScBFM_AB_Type::getMonomerPositionInZ( T_Id i ){ return mPolymerSystem->host[i].z; }

void UpdaterGPUScBFM_AB_Type::setConnectivity
(
    T_Id const iMonomer1,
    T_Id const iMonomer2
)
{
    /* @todo add check whether the bond already exists */
    /* Could also add the inversio, but the bonds are a non-directional graph */
    auto const iNew = mNeighbors->host[ iMonomer1 ].size++;
    if ( iNew > MAX_CONNECTIVITY-1 )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setConnectivity" << "] "
            << "The maximum amount of bonds per monomer (" << MAX_CONNECTIVITY
            << ") has been exceeded!\n";
        throw std::invalid_argument( msg.str() );
    }
    mNeighbors->host[ iMonomer1 ].neighborIds[ iNew ] = iMonomer2;
}

void UpdaterGPUScBFM_AB_Type::setLatticeSize
(
    T_BoxSize const boxX,
    T_BoxSize const boxY,
    T_BoxSize const boxZ
)
{
    if ( mBoxX == boxX && mBoxY == boxY && mBoxZ == boxZ )
        return;

    if ( ! ( inRange< T_Coordinate >( boxX ) &&
             inRange< T_Coordinate >( boxY ) &&
             inRange< T_Coordinate >( boxZ )    ) )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::setLatticeSize" << "] "
            << "The box size (" << boxX << "," << boxY << "," << boxZ
            << ") is larger than the internal integer data type for "
            << "representing positions allow! (" << std::numeric_limits< T_Coordinate >::min()
            << " <= size <= " << std::numeric_limits< T_Coordinate >::max() << ")";
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
    mBoxXLog2  = 0; auto dummy = mBoxX ; while ( dummy >>= 1 ) ++mBoxXLog2;
    mBoxXYLog2 = 0; dummy = mBoxX*mBoxY; while ( dummy >>= 1 ) ++mBoxXYLog2;
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
}

/**
 * Uses mPolymerSystemSortedOld and mPolymerSystemSorted and finds overflows
 * assuming a given physical known maximum movement since the old data.
 * Both inputs are assumed to be on the gpu!
 * If so, then the overflow is reversed if it happened because of data type
 * overflow and/or is counted into mviPolymerSystemSortedVirtualBox if it
 * happened because of the periodic boundary condition of the box.
 */
void UpdaterGPUScBFM_AB_Type::findAndRemoveOverflows( bool copyToHost )
{
    /**
     * find jumps and "deapply" them. We just have to find jumps larger than
     * the number of time steps calculated assuming the monomers can only move
     * 1 cell per time step per direction (meaning this also works with
     * diagonal moves!)
     * If for example the box size is 32, but we also calculate with uint8,
     * then the particles may seemingly move not only bei +-32, but also by
     * +-256, but in both cases the particle actually only moves one virtual
     * box.
     * E.g. the particle was at 0 moved to -1 which was mapped to 255 because
     * uint8 overflowed, but the box size is 64, then deltaMove=255 and we
     * need to subtract 3*64. This only works if the box size is a multiple of
     * the type maximum number (256). I.e. in any sane environment if the box
     * size is a power of two, which was a requirement anyway already.
     * Actually, as the position is just calculated as +-1 without any wrapping,
     * the only way for jumps to happen is because of type overflows.
     */

#if defined( USE_GPU_FOR_OVERHEAD )
    auto const nThreads = 128;
    auto const nBlocks  = ceilDiv( mnMonomersPadded, nThreads );
    /* the padding values do not change, so we can simply let the threads
     * calculate them without worries and save the loop over the species */
    kernelTreatOverflows<<< nBlocks, nThreads, 0, mStream >>>(
        mPolymerSystemSortedOld         ->gpu,
        mPolymerSystemSorted            ->gpu,
        mviPolymerSystemSortedVirtualBox->gpu,
        mnMonomersPadded
    );
    if ( copyToHost )
    {
        mPolymerSystemSorted            ->pop();
        mviPolymerSystemSortedVirtualBox->pop();
    }
#else
    mPolymerSystemSorted            ->popAsync();
    mviPolymerSystemSortedVirtualBox->popAsync();
    CUDA_ERROR( cudaStreamSynchronize( mStream ) );

    size_t nPrintInfo = 10;
    for ( T_Id i = 0u; i < mnAllMonomers; ++i )
    {
        auto const r0tmp = mPolymerSystemSortedOld         ->host[ miToiNew->host[i] ];
        auto const r1tmp = mPolymerSystemSorted            ->host[ miToiNew->host[i] ];
        auto const ivtmp = mviPolymerSystemSortedVirtualBox->host[ miToiNew->host[i] ];
        T_UCoordinateCuda r0[3] = { r0tmp.x, r0tmp.y, r0tmp.z };
        T_UCoordinateCuda r1[3] = { r1tmp.x, r1tmp.y, r1tmp.z };
        T_Coordinate      iv[3] = { ivtmp.x, ivtmp.y, ivtmp.z };

        std::vector< T_BoxSize > const boxSizes = { mBoxX, mBoxY, mBoxZ };
        auto constexpr boxSizeCudaType = 1ll << ( sizeof( T_UCoordinateCuda ) * CHAR_BIT );
        for ( auto iCoord = 0u; iCoord < 3u; ++iCoord )
        {
            assert( boxSizeCudaType >= boxSizes[ iCoord ] );
            //assert( nMonteCarloSteps <= boxSizeCudaType / 2 );
            //assert( nMonteCarloSteps <= std::min( std::min( mBoxX, mBoxY ), mBoxZ ) / 2 );
            auto const deltaMove = r1[ iCoord ] - r0[ iCoord ];
            if ( std::abs( deltaMove ) > boxSizeCudaType / 2 )
            {
                if ( nPrintInfo > 0 )
                {
                    --nPrintInfo;
                    mLog( "Info" )
                        << i << " " << char( 'x' + iCoord ) << ": "
                        << (int) r0[ iCoord ] << " -> " << (int) r1[ iCoord ] << " -> "
                        << T_Coordinate( r1[ iCoord ] - ( boxSizeCudaType - boxSizes[ iCoord ] ) )
                        << "\n";
                }
                r1[ iCoord ] -= boxSizeCudaType - boxSizes[ iCoord ];
                iv[ iCoord ] -= deltaMove > decltype(deltaMove)(0) ? 1 : -1;
            }
        }
        mPolymerSystemSorted->host[ miToiNew->host[i] ].x = r1[0];
        mPolymerSystemSorted->host[ miToiNew->host[i] ].y = r1[1];
        mPolymerSystemSorted->host[ miToiNew->host[i] ].z = r1[2];
        mviPolymerSystemSortedVirtualBox->host[ miToiNew->host[i] ].x = iv[0];
        mviPolymerSystemSortedVirtualBox->host[ miToiNew->host[i] ].y = iv[1];
        mviPolymerSystemSortedVirtualBox->host[ miToiNew->host[i] ].z = iv[2];
    }
    mviPolymerSystemSortedVirtualBox->pushAsync();
#endif
}

/**
 * Checks for excluded volume condition and for correctness of all monomer bonds
 * Beware, it useses and thereby thrashes mLattice. Might be cleaner to declare
 * as const and malloc and free some temporary buffer, but the time ...
 * https://randomascii.wordpress.com/2014/12/10/hidden-costs-of-memory-allocation/
 * "In my tests, for sizes ranging from 8 MB to 32 MB, the cost for a new[]/delete[] pair averaged about 7.5 μs (microseconds), split into ~5.0 μs for the allocation and ~2.5 μs for the free."
 *  => ~40k cycles
 */
void UpdaterGPUScBFM_AB_Type::checkSystem() const
{
    if ( ! mLog.isActive( "Check" ) )
        return;

    /* note that std::vector< bool > already uses bitpacking!
     * We'd have to watch out when erasing that array with memset! */
    std::vector< uint8_t > lattice( mBoxX * mBoxY * mBoxZ, 0 );

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
    for ( T_Id i = 0; i < mnAllMonomers; ++i )
    {
        int32_t const & x = mPolymerSystem->host[i].x;
        int32_t const & y = mPolymerSystem->host[i].y;
        int32_t const & z = mPolymerSystem->host[i].z;
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
        lattice[ linearizeBoxVectorIndex( x  , y  , z   ) ] = 1; /* 0 */
        lattice[ linearizeBoxVectorIndex( x+1, y  , z   ) ] = 1; /* 1 */
        lattice[ linearizeBoxVectorIndex( x  , y+1, z   ) ] = 1; /* 2 */
        lattice[ linearizeBoxVectorIndex( x+1, y+1, z   ) ] = 1; /* 3 */
        lattice[ linearizeBoxVectorIndex( x  , y  , z+1 ) ] = 1; /* 4 */
        lattice[ linearizeBoxVectorIndex( x+1, y  , z+1 ) ] = 1; /* 5 */
        lattice[ linearizeBoxVectorIndex( x  , y+1, z+1 ) ] = 1; /* 6 */
        lattice[ linearizeBoxVectorIndex( x+1, y+1, z+1 ) ] = 1; /* 7 */
    }
    /* check total occupied cells inside lattice to ensure that the above
     * transfer went without problems. Note that the number will be smaller
     * if some monomers overlap!
     * Could also simply reduce mLattice with +, I think, because it only
     * cotains 0 or 1 ??? */
    unsigned nOccupied = 0;
    for ( uint32_t i = 0u; i < mBoxX * mBoxY * mBoxZ; ++i )
        nOccupied += lattice[i] != 0;
    if ( ! ( nOccupied == mnAllMonomers * 8 ) )
    {
        std::stringstream msg;
        msg << "[" << __FILENAME__ << "::~checkSystem" << "] "
            << "Occupation count in mLattice is wrong! Expected 8*nMonomers="
            << 8 * mnAllMonomers << " occupied cells, but got " << nOccupied;
        throw std::runtime_error( msg.str() );
    }

    /**
     * Check bonds i.e. that |dx|<=3 and whether it is allowed by the given
     * bond set
     */
    for ( T_Id i = 0; i < mnAllMonomers; ++i )
    for ( unsigned iNeighbor = 0; iNeighbor < mNeighbors->host[i].size; ++iNeighbor )
    {
        /* calculate the bond vector between the neighbor and this particle
         * neighbor - particle = ( dx, dy, dz ) */
        auto const neighbor = mPolymerSystem->host[ mNeighbors->host[i].neighborIds[ iNeighbor ] ];
        auto dx = (int) neighbor.x - (int) mPolymerSystem->host[i].x;
        auto dy = (int) neighbor.y - (int) mPolymerSystem->host[i].y;
        auto dz = (int) neighbor.z - (int) mPolymerSystem->host[i].z;
        /* with this uncommented, we can ignore if a monomer jumps over the
         * whole box range or T_UCoordinateCuda range */
        /*
        #ifndef NDEBUG
            auto constexpr nLongestBond = 8u;
            assert( mBoxX >= nLongestBond );
            assert( mBoxY >= nLongestBond );
            assert( mBoxZ >= nLongestBond );
        #endif
        dx %= mBoxX; if ( dx < -int( mBoxX )/ 2 ) dx += mBoxX; if ( dx > (int) mBoxX / 2 ) dx -= mBoxX;
        dy %= mBoxY; if ( dy < -int( mBoxY )/ 2 ) dy += mBoxY; if ( dy > (int) mBoxY / 2 ) dy -= mBoxY;
        dz %= mBoxZ; if ( dz < -int( mBoxZ )/ 2 ) dz += mBoxZ; if ( dz > (int) mBoxZ / 2 ) dz -= mBoxZ;
        */

        int erroneousAxis = -1;
        if ( ! ( -3 <= dx && dx <= 3 ) ) erroneousAxis = 0;
        if ( ! ( -3 <= dy && dy <= 3 ) ) erroneousAxis = 1;
        if ( ! ( -3 <= dz && dz <= 3 ) ) erroneousAxis = 2;
        if ( erroneousAxis >= 0 || mForbiddenBonds[ linearizeBondVectorIndex( dx, dy, dz ) ] )
        {
            std::stringstream msg;
            msg << "[" << __FILENAME__ << "::checkSystem] ";
            if ( erroneousAxis > 0 )
                msg << "Invalid " << char( 'X' + erroneousAxis ) << "-Bond: ";
            if ( mForbiddenBonds[ linearizeBondVectorIndex( dx, dy, dz ) ] )
                msg << "This particular bond is forbidden: ";
            msg << "(" << dx << "," << dy<< "," << dz << ") between monomer "
                << i << " at (" << mPolymerSystem->host[i].x << ","
                                << mPolymerSystem->host[i].y << ","
                                << mPolymerSystem->host[i].z << ") and monomer "
                << mNeighbors->host[i].neighborIds[ iNeighbor ] << " at ("
                << neighbor.x << "," << neighbor.y << "," << neighbor.z << ")"
                << std::endl;
             throw std::runtime_error( msg.str() );
        }
    }
}

void UpdaterGPUScBFM_AB_Type::runSimulationOnGPU
(
    uint32_t const nMonteCarloSteps
)
{
    std::clock_t const t0 = std::clock();

    CUDA_ERROR( cudaStreamSynchronize( mStream ) ); // finish e.g. initializations
    CUDA_ERROR( cudaMemcpy( mPolymerSystemSortedOld->gpu, mPolymerSystemSorted->gpu, mPolymerSystemSortedOld->nBytes, cudaMemcpyDeviceToDevice ) );
    auto const nSpecies = mnElementsInGroup.size();
    #if defined( USE_DOUBLE_BUFFERED_TMP_LATTICE )
        cudaStream_t streamMemset = 0;
    #endif

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
        sums .resize( nSpecies, std::vector< double >( nFilters, 0             ) );
        sums2.resize( nSpecies, std::vector< double >( nFilters, 0             ) );
        mins .resize( nSpecies, std::vector< double >( nFilters, mnAllMonomers ) );
        maxs .resize( nSpecies, std::vector< double >( nFilters, 0             ) );
        ns   .resize( nSpecies, std::vector< double >( nFilters, 0             ) );
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

    cudaEvent_t tSort0, tSort1;
    if ( mLog.isActive( "Benchmark" ) )
    {
        cudaEventCreate( &tSort0 );
        cudaEventCreate( &tSort1 );
    }

    /* run simulation */
    for ( uint32_t iStep = 0; iStep < nMonteCarloSteps; ++iStep, ++mAge )
    {
        #if defined( USE_PERIODIC_MONOMER_SORTING )
        if ( mAge % mnStepsBetweenSortings == 0 )
        {
            mLog( "Info" ) << "Resorting at age / step " << mAge << "\n";
            if ( mLog.isActive( "Benchmark" ) )
                cudaEventRecord( tSort0, mStream );

            doSpatialSorting();

            if ( mLog.isActive( "Benchmark" ) )
            {
                cudaEventRecord( tSort1, mStream );
                cudaEventSynchronize( tSort1 );  // basically a StreamSynchronize
                float milliseconds = 0;
                cudaEventElapsedTime( & milliseconds, tSort0, tSort1 );
                std::stringstream sBuffered;
                sBuffered << "tSort = " << milliseconds / 1000. << "s\n";
                mLog( "Benchmark" ) << sBuffered.str();
            }
        }
        #endif

        #if defined( USE_UINT8_POSITIONS )
            /**
             * for uint8_t we have to check for overflows every 127 steps, as
             * for 128 steps we couldn't say whether it actually moved 128 steps
             * or whether it moved 128 steps in the other direction and was wrapped
             * to be equal to the hypothetical monomer above
             */
            auto constexpr boxSizeCudaType = 1ll << ( sizeof( T_UCoordinateCuda ) * CHAR_BIT );
            auto constexpr nStepsBetweenOverflowChecks = boxSizeCudaType / 2 - 1;
            if ( iStep != 0 && iStep % nStepsBetweenOverflowChecks == 0 )
            {
                findAndRemoveOverflows( false );
                CUDA_ERROR( cudaMemcpyAsync( mPolymerSystemSortedOld->gpu,
                    mPolymerSystemSorted->gpu, mPolymerSystemSortedOld->nBytes,
                    cudaMemcpyDeviceToDevice, mStream ) );
            }
        #endif
        /* one Monte-Carlo step:
         *  - tries to move on average all particles one time
         *  - each particle could be touched, not just one group */
        for ( uint32_t iSubStep = 0; iSubStep < nSpecies; ++iSubStep )
        {
            #if defined( USE_NBUFFERED_TMP_LATTICE )
                auto const iStepTotal = iStep * nSpecies + iSubStep;
                auto const iOffsetLatticeTmp = ( iStepTotal % mnLatticeTmpBuffers )
                    * ( mBoxX * mBoxY * mBoxZ * sizeof( mLatticeTmp->gpu[0] )
                    #if defined( USE_BIT_PACKING_TMP_LATTICE )
                        / CHAR_BIT
                    #endif
                );
                auto const texLatticeTmp = mvtLatticeTmp[ iStepTotal % mnLatticeTmpBuffers ];
            #else
                auto const iOffsetLatticeTmp = 0u;
                auto const texLatticeTmp = mLatticeTmp->texture;
            #endif

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
                reinterpret_cast< T_CoordinatesCuda * >( mPolymerSystemSorted->gpu ),
                mPolymerFlags->gpu + mviSubGroupOffsets[ iSpecies ],
                mviSubGroupOffsets[ iSpecies ],
                mLatticeTmp->gpu + iOffsetLatticeTmp,
                mNeighborsSorted->gpu + mNeighborsSortedInfo.getMatrixOffsetElements( iSpecies ),
                mNeighborsSortedInfo.getMatrixPitchElements( iSpecies ),
                mNeighborsSortedSizes->gpu + mviSubGroupOffsets[ iSpecies ],
                mnElementsInGroup[ iSpecies ], seed,
                mLatticeOut->texture
            );

            if ( mLog.isActive( "Stats" ) )
            {
                kernelCountFilteredCheck
                <<< nBlocks, nThreads, 0, mStream >>>(
                    reinterpret_cast< T_CoordinatesCuda * >( mPolymerSystemSorted->gpu ),
                    mPolymerFlags->gpu,
                    mviSubGroupOffsets[ iSpecies ],
                    mLatticeTmp->gpu + iOffsetLatticeTmp,
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
                    reinterpret_cast< T_CoordinatesCuda * >( mPolymerSystemSorted->gpu + mviSubGroupOffsets[ iSpecies ] ),
                    mPolymerFlags->gpu + mviSubGroupOffsets[ iSpecies ],
                    mLatticeOut->gpu,
                    mnElementsInGroup[ iSpecies ],
                    texLatticeTmp
                );
            }
            else
            {
                kernelSimulationScBFMPerformSpecies
                <<< nBlocks, nThreads, 0, mStream >>>(
                    reinterpret_cast< T_CoordinatesCuda * >( mPolymerSystemSorted->gpu + mviSubGroupOffsets[ iSpecies ] ),
                    mPolymerFlags->gpu + mviSubGroupOffsets[ iSpecies ],
                    mLatticeOut->gpu,
                    mnElementsInGroup[ iSpecies ],
                    texLatticeTmp
                );
            }

            if ( mLog.isActive( "Stats" ) )
            {
                kernelCountFilteredPerform
                <<< nBlocks, nThreads, 0, mStream >>>(
                    reinterpret_cast< T_CoordinatesCuda * >( mPolymerSystemSorted->gpu + mviSubGroupOffsets[ iSpecies ] ),
                    mPolymerFlags->gpu + mviSubGroupOffsets[ iSpecies ],
                    mLatticeOut->gpu,
                    mnElementsInGroup[ iSpecies ],
                    texLatticeTmp,
                    dpFiltered
                );
            }

            if ( useCudaMemset )
            {
                #if defined( USE_NBUFFERED_TMP_LATTICE )
                    /* we only need to delete when buffers will wrap around and
                     * on the last loop, so that on next runSimulationOnGPU
                     * call mLatticeTmp is clean */
                    if ( ( iStepTotal % mnLatticeTmpBuffers == 0 ) ||
                         ( iStep == nMonteCarloSteps-1 && iSubStep == nSpecies-1 ) )
                    {
                        cudaMemsetAsync( (void*) mLatticeTmp->gpu, 0, mLatticeTmp->nBytes, mStream );
                    }
                #elif defined( USE_THRUST_FILL )
                    thrust::fill( thrust::system::cuda::par, (uint64_t*)  mLatticeTmp->gpu,
                                  (uint64_t*)( mLatticeTmp->gpu + mLatticeTmp->nElements ), 0 );
                #else
                    #ifdef USE_BIT_PACKING_TMP_LATTICE
                        auto const nBytesToDelete = mLatticeTmp->nBytes / CHAR_BIT;
                    #else
                        auto const nBytesToDelete = mLatticeTmp->nBytes;
                    #endif
                    #ifdef USE_DOUBLE_BUFFERED_TMP_LATTICE
                        /* wait for calculations to finish before we can delete */
                        CUDA_ERROR( cudaStreamSynchronize( mStream ) );
                        /* delete the tmp buffer we have used in the last step */
                        if ( streamMemset == 0 )
                            CUDA_ERROR( cudaStreamCreate( & streamMemset ) )
                        #if 0
                            cudaMemsetAsync( (void*) mLatticeTmp->gpu, 0, nBytesToDelete, streamMemset );
                        #else
                            cudaMemcpyAsync( (void*) mLatticeTmp->gpu, (void*) mLatticeTmp->host, nBytesToDelete, cudaMemcpyHostToDevice, streamMemset );
                        #endif
                        std::swap( mLatticeTmp, mLatticeTmp2 );
                    #else
                        mLatticeTmp->memsetAsync(0);
                    #endif
                #endif
            }
            else
            {
                kernelSimulationScBFMZeroArraySpecies
                <<< nBlocks, nThreads, 0, mStream >>>(
                    reinterpret_cast< T_CoordinatesCuda * >( mPolymerSystemSorted->gpu + mviSubGroupOffsets[ iSpecies ] ),
                    mPolymerFlags->gpu + mviSubGroupOffsets[ iSpecies ],
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
        std::stringstream sBuffered;
        sBuffered << "tGpuLoop = " << milliseconds / 1000. << "s\n";
        mLog( "Benchmark" ) << sBuffered.str();
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

    doCopyBack();

    if ( mLog.isActive( "Benchmark" ) )
    {
        std::clock_t const t1 = std::clock();
        double const dt = float(t1-t0) / CLOCKS_PER_SEC;
        mLog( "Benchmark" )
        << "run time (GPU): " << nMonteCarloSteps << "\n"
        << "mcs = " << nMonteCarloSteps  << "  speed [performed monomer try and move/s] = MCS*N/t: "
        << nMonteCarloSteps * ( mnAllMonomers / dt )  << "     runtime[s]:" << dt << "\n";
    }
}

void UpdaterGPUScBFM_AB_Type::doCopyBack()
{
    mtCopyBack0 = std::chrono::high_resolution_clock::now();

    /* all MCS are done- copy information back from GPU to host */
    if ( mLog.isActive( "Check" ) )
    {
        mLatticeTmp->pop( false ); // sync
        size_t nOccupied = 0;
        for ( size_t i = 0u; i < mLatticeTmp->nElements; ++i )
            nOccupied += mLatticeTmp->host[i] != 0;
        if ( nOccupied != 0 )
        {
            std::stringstream msg;
            msg << "latticeTmp occupation (" << nOccupied << ") should be 0! Exiting ...\n";
            throw std::runtime_error( msg.str() );
        }
    }

    #if defined( USE_PERIODIC_MONOMER_SORTING ) && ! defined( USE_GPU_FOR_OVERHEAD )
        miNewToi->popAsync();
        miToiNew->popAsync(); /* needed for findAndRemoveOverflows, but only if USE_GPU_FOR_OVERHEAD not set */
        CUDA_ERROR( cudaStreamSynchronize( mStream ) );
        checkMonomerReorderMapping();
    #endif

    #if defined( USE_UINT8_POSITIONS )
        cudaEvent_t tOverflowCheck0, tOverflowCheck1;
        if ( mLog.isActive( "Benchmark" ) )
        {
            cudaEventCreate( &tOverflowCheck0 );
            cudaEventCreate( &tOverflowCheck1 );
            cudaEventRecord( tOverflowCheck0, mStream );
        }

        findAndRemoveOverflows( false );

        if ( mLog.isActive( "Benchmark" ) )
        {
            // https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc/#disqus_thread
            cudaEventRecord( tOverflowCheck1, mStream );
            cudaEventSynchronize( tOverflowCheck1 );  // basically a StreamSynchronize
            float milliseconds = 0;
            cudaEventElapsedTime( & milliseconds, tOverflowCheck0, tOverflowCheck1 );
            std::stringstream sBuffered;
            sBuffered << "tOverflowCheck = " << milliseconds / 1000. << "s\n";
            mLog( "Benchmark" ) << sBuffered.str();
        }
    #endif

#if defined( USE_GPU_FOR_OVERHEAD )
    auto const nThreads = 128;
    auto const nBlocksP = ceilDiv( mnMonomersPadded, nThreads );
    kernelUndoPolymerSystemSorting<<< nBlocksP, nThreads, 0, mStream >>>
    (
        mPolymerSystemSorted            ->gpu,
        mviPolymerSystemSortedVirtualBox->gpu,
        miNewToi                        ->gpu,
        mPolymerSystem                  ->gpu,
        mnMonomersPadded
    );
    mPolymerSystem->pop();
#else
    mPolymerSystemSorted->pop();
    mviPolymerSystemSortedVirtualBox->pop();
    /* untangle reordered array so that LeMonADE can use it again */
    for ( T_Id i = 0u; i < mnAllMonomers; ++i )
    {
        auto const pTarget = mPolymerSystemSorted->host + miToiNew->host[i];
        if ( i < 10 )
            mLog( "Info" ) << "Copying back " << i << " from " << miToiNew->host[i] << "\n";
        mPolymerSystem->host[i].x = pTarget->x + mviPolymerSystemSortedVirtualBox->host[ miToiNew->host[i] ].x * mBoxX;
        mPolymerSystem->host[i].y = pTarget->y + mviPolymerSystemSortedVirtualBox->host[ miToiNew->host[i] ].y * mBoxY;
        mPolymerSystem->host[i].z = pTarget->z + mviPolymerSystemSortedVirtualBox->host[ miToiNew->host[i] ].z * mBoxZ;
        mPolymerSystem->host[i].w = pTarget->w;
    }
#endif

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
    this->destruct();

    cudaDeviceSynchronize();
    cudaProfilerStop();
}
