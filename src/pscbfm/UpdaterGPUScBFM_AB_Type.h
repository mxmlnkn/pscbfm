/*
 * UpdaterGPUScBFM_AB_Type.h
 *
 *  Created on: 27.07.2017
 *      Author: Ron Dockhorn
 */

#pragma once


#include <cassert>
#include <chrono>                           // high_resolution_clock
#include <cstdio>                           // printf
#include <cstdint>                          // uint32_t, size_t
#include <stdexcept>

#include <cuda_runtime_api.h>               // cudaStream_t
#include <LeMonADE/utility/RandomNumberGenerators.h>

#include "cudacommon.hpp"
#include "SelectiveLogger.hpp"



/* This is still used, at least the 32 bit version! */
#if 0
    typedef uint32_t uintCUDA;
    typedef int32_t  intCUDA;
    #define MASK5BITS 0x7FFFFFE0
#else
    typedef uint16_t uintCUDA;
    typedef int16_t  intCUDA;
    #define MASK5BITS 0x7FE0
#endif


#define MAX_CONNECTIVITY 4 // original connectivity
// #define MAX_CONNECTIVITY 8 // needed for the coloring example

//#define NONPERIODICITY


class UpdaterGPUScBFM_AB_Type
{
private:
    SelectedLogger mLog;

    cudaStream_t mStream;

    RandomNumberGenerators randomNumbers;

    bool mForbiddenBonds[512];
    //int BondAsciiArray[512];

    uint32_t nAllMonomers;

    /**
     * Vector of length boxX * boxY * boxZ. Actually only contains 0 if
     * the cell / lattice point is empty or 1 if it is occupied by a monomer
     * Suggestion: bitpack it to save 8 times memory and possibly make the
     *             the reading faster if it is memory bound ???
     */
    uint8_t * mLattice; // being used for checkLattice nothing else ...
    MirroredTexture< uint8_t > * mLatticeOut, * mLatticeTmp;

    /* copy into mPolymerSystem and drop the property tag while doing so.
     * would be easier and probably more efficient if mPolymerSystem_device/host
     * would be a struct of arrays instead of an array of structs !!! */
    /**
     * Contains the nMonomers particles as well as a property tag for each:
     *   [ x0, y0, z0, p0, x1, y1, z1, p1, ... ]
     * The property tags p are bit packed:
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
     * The saved location is used as the lower left front corner when
     * populating the lattice with 2x2x2 boxes representing the monomers
     */
    std::vector< intCUDA > mPolymerSystem;
    MirroredVector< intCUDA > * mPolymerSystemSorted;

    static auto constexpr nBytesAlignment    = 512u;
    static auto constexpr nElementsAlignment = nBytesAlignment / ( 4u * sizeof( intCUDA ) );
    static_assert( nBytesAlignment == nElementsAlignment * 4u * sizeof( intCUDA), "Element type of polymer systems seems to be larger than the Bytes we need to align on!" );

    /* for each monomer the attribute 1 (A) or 2 (B) is stored
     * -> could be 1 bit per monomer ... !!!
     * @see http://www.gotw.ca/gotw/050.htm
     * -> wow std::vector<bool> already optimized for space with bit masking!
     * This is only needed once for initializing mMonomerIdsA,B */
    int32_t * mAttributeSystem;
    std::vector< uint8_t > mGroupIds; /* for each monomer stores the color / attribute / group ID/tag */
    std::vector< size_t > mnElementsInGroup;
    std::vector< size_t > iToiNew;   /* for each old monomer stores the new position */
    std::vector< size_t > iNewToi;   /* for each new monomer stores the old position */
    std::vector< size_t > iSubGroupOffset; /* stores offsets (in number of elements not bytes) to each aligned subgroup vector in mPolymerSystemSorted */

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
        uint32_t size;
        uint32_t neighborIds[ MAX_CONNECTIVITY ];
    };
    /* size is encoded in mPolymerSystem to make things faster */
    struct MonomerEdgesCompressed
    {
        uint32_t neighborIds[ MAX_CONNECTIVITY ];
    };
private:
    std::vector< MonomerEdges > mNeighbors;
    /**
     * stores the IDs of all neighbors as is needed to check for the bond
     * set / length restrictions.
     * But after the sorting of mPolymerSystem the IDs also changed.
     * And as I don't want to push iToiNew and iNewToi to the GPU instead
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
    MirroredVector< uint32_t > * mNeighborsSorted;
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

    AlignedMatrices< uint32_t > mNeighborsSortedInfo;

    uint32_t   mBoxX     ;
    uint32_t   mBoxY     ;
    uint32_t   mBoxZ     ;
    uint32_t   mBoxXM1   ;
    uint32_t   mBoxYM1   ;
    uint32_t   mBoxZM1   ;
    uint32_t   mBoxXLog2 ;
    uint32_t   mBoxXYLog2;

    uint32_t linearizeBoxVectorIndex
    (
        uint32_t const & ix,
        uint32_t const & iy,
        uint32_t const & iz
    );

    /**
     * Checks for excluded volume condition and for correctness of all monomer bonds
     */
    void checkSystem();

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

    void initialize();
    inline bool execute(){ return true; }
    void cleanup();

    /* setter methods */
    void setGpu               ( int iGpuToUse );
    void copyBondSet( int dx, int dy, int dz, bool bondForbidden );
    void setNrOfAllMonomers   ( uint32_t nAllMonomers );
    void setAttribute         ( uint32_t i, int32_t attribute );
    void setMonomerCoordinates( uint32_t i, int32_t x, int32_t y, int32_t z );
    void setConnectivity      ( uint32_t monoidx1, uint32_t monoidx2 );
    void setLatticeSize       ( uint32_t boxX, uint32_t boxY, uint32_t boxZ );

    /**
     * sets monomer positions given in mPolymerSystem in mLattice to occupied
     */
    void populateLattice();
    void runSimulationOnGPU( int32_t nrMCS_per_Call );

    int32_t getMonomerPositionInX( uint32_t i );
    int32_t getMonomerPositionInY( uint32_t i );
    int32_t getMonomerPositionInZ( uint32_t i );

    void setPeriodicity( bool isPeriodicX, bool isPeriodicY, bool isPeriodicZ );

    /* for benchmarking purposes */
    std::chrono::time_point< std::chrono::high_resolution_clock > mtCopyBack0;
};
