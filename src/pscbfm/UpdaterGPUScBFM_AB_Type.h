/*
 * UpdaterGPUScBFM_AB_Type.h
 *
 *  Created on: 27.07.2017
 *      Authors: Ron Dockhorn, Maximilian Knespel
 */

#pragma once


#include <cassert>
#include <chrono>                           // high_resolution_clock
#include <cstdio>                           // printf
#include <cstdint>                          // uint32_t, size_t
#include <stdexcept>
#include <type_traits>                      // make_unsigned

#include <cuda_runtime_api.h>               // cudaStream_t, cudaDeviceProp
#include <LeMonADE/utility/RandomNumberGenerators.h>

#include "cudacommon.hpp"
#include "SelectiveLogger.hpp"



/**
 * When not reordering the neighbor information as struct of array,
 * then increasing this leads to performance degradataion!
 * But currently, as the reordering is implemented, it just leads to
 * higher memory usage.
 * In the 3D case more than 20 makes no sense for the standard bond vector
 * set, as the volume exclusion plus the bond vector set make 20 neighbors
 * the maximum. In real use cases 8 are already very much / more than sufficient.
 */
#define MAX_CONNECTIVITY 8

//#define NONPERIODICITY


/**
 * For each species this stores the
 *  - max. number of neighbors
 *  - location in memory for the species edge matrix
 *  - number of elements / monomers
 *  - pitched number of monomers / bytes to get alignment
 */
template< typename T >
struct AlignedMatrices
{
    using value_type = T;
    size_t mnBytesAlignment;
    /* bytes one row takes up ( >= nCols * sizeof(T) ) */
    struct MatrixMemoryInfo {
        size_t nRows, nCols, iOffsetBytes, nBytesPitch;
    };
    std::vector< MatrixMemoryInfo > mMatrices;

    inline AlignedMatrices( size_t rnBytesAlignment = 512u )
     : mnBytesAlignment( rnBytesAlignment )
    {
        /* this simplifies some "recursive" calculations */
        MatrixMemoryInfo m;
        m.nRows        = 0;
        m.nCols        = 0;
        m.iOffsetBytes = 0;
        m.nBytesPitch  = 0;
        mMatrices.push_back( m );
    }

    inline void newMatrix
    (
        size_t const nRows,
        size_t const nCols
    )
    {
        auto const & l = mMatrices[ mMatrices.size()-1 ];
        MatrixMemoryInfo m;
        m.nRows        = nRows;
        m.nCols        = nCols;
        m.iOffsetBytes = l.iOffsetBytes + l.nRows * l.nBytesPitch;
        m.nBytesPitch  = ceilDiv( nCols * sizeof(T), mnBytesAlignment ) * mnBytesAlignment;
        mMatrices.push_back( m );
    }

    inline size_t getMatrixOffsetBytes( size_t const iMatrix ) const
    {
        /* 1+ because of dummy 0-th element */
        return mMatrices.at( 1+iMatrix ).iOffsetBytes;
    }
    inline size_t getMatrixPitchBytes( size_t const iMatrix ) const
    {
        return mMatrices.at( 1+iMatrix ).nBytesPitch;
    }
    inline size_t getRequiredBytes( void ) const
    {
        auto const & l = mMatrices[ mMatrices.size()-1 ];
        return l.iOffsetBytes + l.nRows * l.nBytesPitch;
    }

    inline size_t bytesToElements( size_t const nBytes ) const
    {
        assert( nBytes / sizeof(T) * sizeof(T) == nBytes );
        return nBytes / sizeof(T);
    }

    inline size_t getMatrixOffsetElements( size_t const iMatrix ) const
    {
        return bytesToElements( getMatrixOffsetBytes( iMatrix ) );
    }
    inline size_t getMatrixPitchElements( size_t const iMatrix ) const
    {
        return bytesToElements( getMatrixPitchBytes( iMatrix ) );
    }
    inline size_t getRequiredElements( void ) const
    {
        return bytesToElements( getRequiredBytes() );
    }
};

class UpdaterGPUScBFM_AB_Type
{
public:
    /**
     * This is still used, at least the 32 bit version!
     * These are the types for the monomer positions.
     * Even for the periodic case, we are sometimes interested in the "global"
     * position without fold back to the periodic box in order to calculate
     * diffusion easily.
     * Although you also could derive an algorithm for automatically finding
     * larger jumps and undo the folding back manually. In order to not miss
     * a case you would have to do this every
     *     dtApplyJumps = ceilDiv( min( boxX, boxY, boxZ ), 2 ) - 1
     * time steps. If a particle moved dtApplyJumps+1, then you could be sure
     * that it was indeed because of the periodicity condition.
     * But then again, using uint8_t would limit the box size to 256 which we
     * do not want. The bit operation introduced limitation to 1024 is already
     * a little bit worrisome.
     * But IF you have such a small box and you can use uint8_t, then we could
     * do as written above and possibly make the algorithm EVEN FASTER as the
     * memory bandwidth could be reduced even more!
     */
    using T_BoxSize          = uint64_t; // uint32_t // should be unsigned!
    using T_Coordinate       = int32_t; // int64_t // should be signed!
    using T_CoordinateCuda   = int8_t; // int32_t (int8_t, uint8_t does not work for a 256^3 box :S ??? )
    using T_UCoordinateCuda  = std::make_unsigned< T_CoordinateCuda >::type;
    using T_Coordinates      = CudaVec4< T_Coordinate      >::value_type;
    using T_CoordinatesCuda  = CudaVec4< T_CoordinateCuda  >::value_type;
    using T_UCoordinatesCuda = CudaVec4< T_UCoordinateCuda >::value_type;
    /* could also be uint8_t if you know you only have 256 different
     * species at maximum. For the autocoloring this is implicitly true,
     * but not so if the user manually specifies colors! */
    using T_Color   = uint32_t;
    using T_Flags   = uint8_t ; // uint16_t, uint32_t
    using T_Id      = uint32_t; // should be unsigned!
    using T_Lattice = uint8_t ; // untested for something else than uint8_t!

private:
    SelectedLogger mLog;

    cudaStream_t mStream;

    RandomNumberGenerators randomNumbers;

    bool mForbiddenBonds[512];
    //int BondAsciiArray[512];

    /**
     * Vector of length boxX * boxY * boxZ. Actually only contains 0 if
     * the cell / lattice point is empty or 1 if it is occupied by a monomer
     * Suggestion: bitpack it to save 8 times memory and possibly make the
     *             the reading faster if it is memory bound ???
     */
    T_Lattice * mLattice; // being used for checkLattice nothing else ...
    MirroredTexture< T_Lattice > * mLatticeOut, * mLatticeTmp, * mLatticeTmp2;
    /**
     * when using bit packing only 1/8 of mLatticeTmp is used. In order to
     * to using everything we can simply increment the mLatticeTmp->gpu pointer,
     * but for textures this is not possible so easily, therefore we need to
     * store texture objects for each bit packed sub lattice in mLatticeTmp.
     * Then after 8 usages we can call one cudaMemset for all, possibly making
     * 8 times better use of parallelism on the GPU!
     */
    cudaTextureObject_t mvtLatticeTmp[ CHAR_BIT ];

    /* copy into mPolymerSystem and drop the property tag while doing so.
     * would be easier and probably more efficient if mPolymerSystem_device/host
     * would be a struct of arrays instead of an array of structs !!! */
    /**
     * Contains the nMonomers particles as well as a property tag for each:
     *   [ x0, y0, z0, p0, x1, y1, z1, p1, ... ]
     * The property tags p are bit packed:
     * @verbatim
     *                        8  7  6  5  4  3  2  1  0
     * +--------+--+--+--+--+--+--+--+--+--+--+--+--+--+
     * | unused |  |  |  |  |c |   nnr  |  dir   |move |
     * +--------+--+--+--+--+--+--+--+--+--+--+--+--+--+
     *  c   ... charged: 0 no, 1: yes
     *  nnr ... number of neighbors, this will get populated from LeMonADE's
     *          get get
     *  move ... Bit 0 set by kernelCheckSpecies if move was found to be possible
     *           Bit 1 set by kernelPerformSpecies if move still possible
     *           heeding the new locations of all particles.
     *           If both bits set, the move gets applied to polymerSystem in
     *           kernelZeroArrays
     * @endverbatim
     * The saved location is used as the lower left front corner when
     * populating the lattice with 2x2x2 boxes representing the monomers
     */
    size_t mnAllMonomers;
    std::vector< T_Coordinate > mPolymerSystem;
    /**
     * This is mPolymerSystem sorted by species and also made struct of array
     * in order to split neighbors size off into extra array, thereby also
     * increasing max neighbor size from 8 to 256!
     * @verbatim
     * A1x A2x A3x A4x ... A1y A2y A3y A4y ... A1z A2z ... B1x B2x ...
     * @endverbatim
     * Note how this struct of array leads to yet another alignment problem
     * I think I need AlignedMatrices for this, too :(
     */
    size_t mnMonomersPadded;
    MirroredVector< T_UCoordinatesCuda > * mPolymerSystemSorted;
    MirroredVector< T_UCoordinatesCuda > * mPolymerSystemSortedOld;
    MirroredVector< T_Coordinates      > * mviPolymerSystemSortedVirtualBox;
    /**
     * These are to be used for storing the flags and chosen direction of
     * the old property tag.
     *      4  3  2  1  0
     *    +--+--+--+--+--+
     *    |  dir   |move |
     *    +--+--+--+--+--+
     * These are currently temporary vectors only written and read to from
     * the GPU, so MirroredVector isn't necessary, but it's easy to use and
     * could be nice for debugging (e.g. to replace the count kernels)
     */
    MirroredVector< T_Flags > * mPolymerFlags;

    static auto constexpr nBytesAlignment    = 512u;
    static auto constexpr nElementsAlignment = nBytesAlignment / ( 4u * sizeof( T_UCoordinateCuda ) );
    static_assert( nBytesAlignment == nElementsAlignment * 4u * sizeof( T_UCoordinateCuda ),
                   "Element type of polymer systems seems to be larger than the Bytes we need to align on!" );

    /* for each monomer the attribute 1 (A) or 2 (B) is stored
     * -> could be 1 bit per monomer ... !!!
     * @see http://www.gotw.ca/gotw/050.htm
     * -> wow std::vector<bool> already optimized for space with bit masking!
     * This is only needed once for initializing mMonomerIdsA,B */
    int32_t * mAttributeSystem;
    std::vector< T_Color > mGroupIds; /* for each monomer stores the color / attribute / group ID/tag */
    std::vector< size_t > mnElementsInGroup;
    std::vector< size_t > miToiNew;   /* for each old monomer stores the new position */
    std::vector< size_t > miNewToi;   /* for each new monomer stores the old position */
    std::vector< size_t > viSubGroupOffsets; /* stores offsets (in number of elements not bytes) to each aligned subgroup vector in mPolymerSystemSorted */

    /* needed to decide whether we can even check autocoloring with given one */
    bool bSetAttributeCalled;
    /* in order to decide how to work with setMonomerCoordinates and
     * getMonomerCoordinates. This way we might just return the monomer
     * info directly instead of needing to desort after each execute run */
    bool bPolymersSorted;

public:
    /* stores amount and IDs of neighbors for each monomer */
    struct MonomerEdges
    {
        T_Id size; // could also be uint8_t as it is limited by MAX_CONNECTIVITY
        T_Id neighborIds[ MAX_CONNECTIVITY ];
    };
    /* size is encoded in mPolymerSystem to make things faster */
    struct MonomerEdgesCompressed
    {
        T_Id neighborIds[ MAX_CONNECTIVITY ];
    };
private:
    std::vector< MonomerEdges > mNeighbors;
    /**
     * stores the IDs of all neighbors as is needed to check for the bond
     * set / length restrictions.
     * But after the sorting of mPolymerSystem the IDs also changed.
     * And as I don't want to push miToiNew and miNewToi to the GPU instead
     * I just change the IDs for all neighbors. Plus the array itself gets
     * rearranged to the new AAA...BBB...CC... ordering
     *
     * In the next level optimization this sorting, i.e. A{neighbor 1234}A...
     * Needs to get also resorted to:
     *
     * @verbatim
     *       nMonomersPaddedInGroup[0] = 8
     *    <---------------------------->
     *    A11 A21 A31 A41 A51  0   0   0
     *    A21 A22 A32 A42 A52  0   0   0
     *    ...
     *    nMonomersPaddedInGroup[1] = 4
     *    <------------>
     * +> B11 B21  0   0
     * |  B12 B22  0   0
     * |  ...
     * +-- at index offset \sum_{i<s} nMonomersPaddedInGroup[i] * MAX_CONNECTIVITY
     *     which is NOT equal to iSubGroupOffset[s] * MAX_CONNECTIVITY if the
     *     padding is different per species!
     * @endverbatim
     *
     * where Aij denotes the j-th neighbor of monomer i, such that parallel
     * access over all monomers of one species to the 1st,2nd,... neighbor
     * is linear in memory.
     * Having A11, A21, B11 aligned instead of only A11,B11,C11,... would be
     * optimal.
     *
     * Because the alignment of the PolymerSystem has more data, its alignment
     * is a harder condition than the alignment of mNeighbors, therefore we
     * don't have to recalculate everything again, and instead use the
     * alignments given as number of elements in iSubGroupOffset.
     * But the offsets are not enough, better would be a species-wise padding
     * number, i.e. nMonomersPaddedInGroup[i] = iSubGroupOffset[s+1] -
     * iSubGroupOffset[s] and the last entry would have to be recalculated
     *   -> just set this in the same loop where iSubGroupOffset is calculated
     *
     * Therefore the access to the j-th neighbor of monomer i of species s
     * would be ... too complicated, I need a new class for this problem.
     */
    MirroredVector < T_Id    > * mNeighborsSorted;
    MirroredVector < uint8_t > * mNeighborsSortedSizes;
    AlignedMatrices< T_Id    >   mNeighborsSortedInfo;

    T_BoxSize mBoxX     ;
    T_BoxSize mBoxY     ;
    T_BoxSize mBoxZ     ;
    T_BoxSize mBoxXM1   ;
    T_BoxSize mBoxYM1   ;
    T_BoxSize mBoxZM1   ;
    T_BoxSize mBoxXLog2 ;
    T_BoxSize mBoxXYLog2;

    int            miGpuToUse;
    cudaDeviceProp mCudaProps;

    /**
     * If we constrict each index to 1024=2^10 which already is quite large,
     * 256=2^8 being normally large, then this means that the linearzed index
     * should have a range of 2^30, meaning uint32_t as output is pretty
     * fixed with uint16_t being way too few bits
     */
    T_Id linearizeBoxVectorIndex
    (
        T_Coordinate const & ix,
        T_Coordinate const & iy,
        T_Coordinate const & iz
    ) const;

    /**
     * Checks for excluded volume condition and for correctness of all monomer bonds
     */
    void checkSystem() const;

public:
    UpdaterGPUScBFM_AB_Type();
    virtual ~UpdaterGPUScBFM_AB_Type();
    void destruct();

    /**
     * all these setter methods are quite finicky in how they are to be used!
     * Dependencies:
     *   setGpu                 : None
     *   copyBondSet            : None
     *   setLatticeSize         : None
     *   setNrOfAllMonomers     : setGpu
     *   setAttribute           : setNrOfAllMonomers
     *   setMonomerCoordinates  : setNrOfAllMonomers
     *   setConnectivity        : setNrOfAllMonomers
     *   initialize             : copyBondSet, setAttribute, setNrOfAllMonomers, setConnectivity, setLatticeSize
     *   execute                : initialize
     * => normally setNrOfAllMonomers and setGpu schould be in the constructor ... :S
     */

private:
    void initializeBondTable();
    void initializeSpeciesSorting(); /* using miNewToi and miToiNew the monomers are mapped to be sorted by species */
    void initializeSpatialSorting(); /* miNewToi and miToiNew will be updated so that monomers are sorted spatially per species */
    void initializeSortedNeighbors();
    void initializeSortedMonomerPositions();
    void initializeLattices();
    void checkMonomerReorderMapping();
    void findAndRemoveOverflows( bool copyToHost = true );
    /**
     * sets monomer positions given in mPolymerSystem in mLattice to occupied
     */
    void populateLattice();

public:
    void initialize();
    inline bool execute(){ return true; }
    void cleanup();

    /* setter methods */
    void setGpu               ( int iGpuToUse );
    void copyBondSet( int dx, int dy, int dz, bool bondForbidden );
    void setNrOfAllMonomers   ( T_Id nAllMonomers );
    void setAttribute         ( T_Id i, int32_t attribute ); // this is to be NOT the coloring as needed for parallelizing the BFM, it is to be used for additional e.g. physical attributes like actual chemical types
    void setMonomerCoordinates( T_Id i, T_Coordinate x, T_Coordinate y, T_Coordinate z );
    void setConnectivity      ( T_Id monoidx1, T_Id monoidx2 );
    void setLatticeSize       ( T_BoxSize boxX, T_BoxSize boxY, T_BoxSize boxZ );

    void runSimulationOnGPU( uint32_t nrMCS_per_Call );

    /* using T_Coordinate with int64_t throws error as LeMonADE itself is limited to 32 bit positions! */
    int32_t getMonomerPositionInX( T_Id i );
    int32_t getMonomerPositionInY( T_Id i );
    int32_t getMonomerPositionInZ( T_Id i );

    void setPeriodicity( bool isPeriodicX, bool isPeriodicY, bool isPeriodicZ );

    /* for benchmarking purposes */
    std::chrono::time_point< std::chrono::high_resolution_clock > mtCopyBack0;
};
