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


// #define MAX_CONNECTIVITY 4 // original connectivity
#define MAX_CONNECTIVITY 8 // needed for the coloring example

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
private:
    std::vector< MonomerEdges > mNeighbors;
    MirroredVector< MonomerEdges > * mNeighborsSorted; /* stores the IDs of all neighbors as is needed to check for bond length restrictions. But after the sorting of mPolymerSystem the IDs also changed. And as I don't want to push iToiNew and iNewToi to the GPU instead I just change the IDs for all neighbors. Plus the array itself gets rearranged to the new AAA...BBB...CC... ordering */

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
