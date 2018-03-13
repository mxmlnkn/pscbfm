#pragma once


#include <chrono>                           // std::chrono::high_resolution_clock
#include <climits>                          // CHAR_BIT
#include <limits>                           // numeric_limits
#include <iostream>

#include <LeMonADE/updater/AbstractUpdater.h>
#include <LeMonADE/utility/Vector3D.h>      // VectorInt3

#include "UpdaterGPUScBFM_AB_Type.h"
#include "SelectiveLogger.hpp"


#define USE_UINT8_POSITIONS


/**
 * Why is this abstraction layer being used, instead of just incorporating
 * the GPU updated into this class?
 * I think because it was tried to get a LeMonADE independent .cu file for
 * the kernels while we still need to inherit from AbstractUpdater
 */


template< class T_IngredientsType >
class GPUScBFM_AB_Type : public AbstractUpdater
{
public:
    typedef T_IngredientsType IngredientsType;
    typedef typename T_IngredientsType::molecules_type MoleculesType;

protected:
    IngredientsType & mIngredients;
    MoleculesType   & molecules   ;

private:
    /**
     * can't use uint8_t for boxes larger 256 on any side, so choose
     * automatically the correct type
     * ... this is some fine stuff. I almost would have wrapped all the
     * method bodies inside macros ... in order to copy paste them inside
     * an if-else-statement
     * But it makes sense, it inherits from all and then type casts it to
     * call the correct methods and members even though all classes we
     * inherit from basically shadow each other
     * @see https://stackoverflow.com/questions/3422106/how-do-i-select-a-member-variable-with-a-type-parameter
     */
    struct WrappedTemplatedUpdaters :
        #define TMP_WRAPPED_UPDATERS_PERIODS(T)            \
        UpdaterGPUScBFM_AB_Type< T, false, false, false >, \
        UpdaterGPUScBFM_AB_Type< T, false, false,  true >, \
        UpdaterGPUScBFM_AB_Type< T, false,  true, false >, \
        UpdaterGPUScBFM_AB_Type< T, false,  true,  true >, \
        UpdaterGPUScBFM_AB_Type< T,  true, false, false >, \
        UpdaterGPUScBFM_AB_Type< T,  true, false,  true >, \
        UpdaterGPUScBFM_AB_Type< T,  true,  true, false >, \
        UpdaterGPUScBFM_AB_Type< T,  true,  true,  true >
        TMP_WRAPPED_UPDATERS_PERIODS( uint8_t  ),
        TMP_WRAPPED_UPDATERS_PERIODS( uint16_t ),
        TMP_WRAPPED_UPDATERS_PERIODS( int16_t  ),
        TMP_WRAPPED_UPDATERS_PERIODS( int32_t  )
        #undef TMP_WRAPPED_UPDATERS_TYPES
    {};
    WrappedTemplatedUpdaters mUpdatersGpu;

    int miGpuToUse;
    //! Number of Monte-Carlo Steps (mcs) to be executed (per GPU-call / Updater call)
    uint32_t mnSteps;
    SelectedLogger mLog;
    bool mCanUseUint8Positions;

protected:
    inline T_IngredientsType & getIngredients() { return mIngredients; }

public:
    /**
     * @brief Standard constructor: initialize the ingredients and specify the GPU.
     *
     * @param rIngredients  A reference to the IngredientsType - mainly the system
     * @param rnSteps       Number of mcs to be executed per GPU-call
     * @param riGpuToUse    ID of the GPU to use. Default: 0
     */
    inline GPUScBFM_AB_Type
    (
        T_IngredientsType & rIngredients,
        uint32_t            rnSteps     ,
        int                 riGpuToUse = 0
    )
    : mIngredients( rIngredients                   ),
      molecules   ( rIngredients.modifyMolecules() ),
      miGpuToUse  ( riGpuToUse                     ),
      mnSteps     ( rnSteps                        ),
      mLog        ( __FILENAME__                   )
    {
        mLog.  activate( "Benchmark" );
        mLog.deactivate( "Check"     );
        mLog.  activate( "Error"     );
        mLog.  activate( "Info"      );
        mLog.  activate( "Stat"      );
        mLog.deactivate( "Warning"   );
    }

    /**
     * Copies required data and parameters from mIngredients to mUpdaterGpu
     * and calls the mUpdaterGpu initializer
     * mIngredients can't just simply be given, because we want to compile
     * UpdaterGPUScBFM_AB_Type.cu by itself and explicit template instantitation
     * over T_IngredientsType is basically impossible
     */
    template< typename T_UCoordinateCuda, bool T_IsPeriodicX, bool T_IsPeriodicY, bool T_IsPeriodicZ >
    inline bool initializeUpdater()
    {
        UpdaterGPUScBFM_AB_Type< T_UCoordinateCuda, T_IsPeriodicX, T_IsPeriodicY, T_IsPeriodicZ > & mUpdaterGpu = mUpdatersGpu;

        mLog( "Info" ) << "Size of mUpdater: " << sizeof( mUpdaterGpu ) << " Byte\n";
        mLog( "Info" ) << "Size of WrappedTemplatedUpdaters: " << sizeof( WrappedTemplatedUpdaters ) << " Byte\n";

        auto const tInit0 = std::chrono::high_resolution_clock::now();

        mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] Forwarding relevant paramters to GPU updater\n";
        mUpdaterGpu.setGpu( miGpuToUse );
        mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] mUpdaterGpu.setPeriodicity\n";
        /* Forward needed parameters to the GPU updater */
        mUpdaterGpu.setAge( mIngredients.modifyMolecules().getAge() );
        mUpdaterGpu.setPeriodicity( mIngredients.isPeriodicX(),
                                    mIngredients.isPeriodicY(),
                                    mIngredients.isPeriodicZ() );

        /* copy monomer positions, attributes and connectivity of all monomers */
        mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] mUpdaterGpu.setLatticeSize\n";
        mUpdaterGpu.setLatticeSize( mIngredients.getBoxX(),
                                    mIngredients.getBoxY(),
                                    mIngredients.getBoxZ() );
        mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] mUpdaterGpu.setNrOfAllMonomers\n";
        mUpdaterGpu.setNrOfAllMonomers( mIngredients.getMolecules().size() );
        mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] mUpdaterGpu.setMonomerCoordinates\n";
        for ( size_t i = 0u; i < mIngredients.getMolecules().size(); ++i )
        {
            mUpdaterGpu.setMonomerCoordinates( i, molecules[i].getX(),
                                                  molecules[i].getY(),
                                                  molecules[i].getZ() );
        }
        mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] mUpdaterGpu.setAttribute\n";
        for ( size_t i = 0u; i < mIngredients.getMolecules().size(); ++i )
            mUpdaterGpu.setAttribute( i, mIngredients.getMolecules()[i].getAttributeTag() );
        mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] mUpdaterGpu.setConnectivity\n";
        for ( size_t i = 0u; i < mIngredients.getMolecules().size(); ++i )
        for ( size_t iBond = 0; iBond < mIngredients.getMolecules().getNumLinks(i); ++iBond )
            mUpdaterGpu.setConnectivity( i, mIngredients.getMolecules().getNeighborIdx( i, iBond ) );

         // false-allowed; true-forbidden
        mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] copy bondset from LeMonADE to GPU-class for BFM\n";
        /* maximum of (expected!!!) bond length in one dimension. Should be
         * queryable or there should be a better way to copy the bond set.
         * Note that supported range is [-4,3] */
        int const maxBondLength = 4;
        for ( int dx = -maxBondLength; dx < maxBondLength; ++dx )
        for ( int dy = -maxBondLength; dy < maxBondLength; ++dy )
        for ( int dz = -maxBondLength; dz < maxBondLength; ++dz )
        {
            /* !!! The negation is confusing, again there should be a better way to copy the bond set */
            mUpdaterGpu.copyBondSet( dx, dy, dz, ! mIngredients.getBondset().isValid( VectorInt3( dx, dy, dz ) ) );
        }

        mLog( "Info" ) << "[" << __FILENAME__ << "::initialize] initialize GPU updater\n";
        mUpdaterGpu.initialize();

        auto const tInit1 = std::chrono::high_resolution_clock::now();
        std::stringstream sBuffered;
        sBuffered << "tInit = " << std::chrono::duration<double>( tInit1 - tInit0 ).count() << "s\n";
        mLog( "Benchmark" ) << sBuffered.str();

        return true;
    }

    /**
     * Was the 'virtual' really necessary ??? I don't think there will ever be
     * some class inheriting from this class...
     * https://en.wikipedia.org/wiki/Virtual_function
     */
    template< typename T_UCoordinateCuda, bool T_IsPeriodicX, bool T_IsPeriodicY, bool T_IsPeriodicZ >
    inline bool executeUpdater()
    {
        UpdaterGPUScBFM_AB_Type< T_UCoordinateCuda, T_IsPeriodicX, T_IsPeriodicY, T_IsPeriodicZ > & mUpdaterGpu = mUpdatersGpu;

        std::clock_t const t0 = std::clock();

        mLog( "Info" ) << "[" << __FILENAME__ << "] MCS:" << mIngredients.getMolecules().getAge() << "\n";
        mLog( "Info" ) << "[" << __FILENAME__ << "] start simulation on GPU\n";

        mUpdaterGpu.setAge( mIngredients.modifyMolecules().getAge() );
        mUpdaterGpu.runSimulationOnGPU( mnSteps ); // sets mtCopyBack0
        auto const tCopyBack0 = mUpdaterGpu.mtCopyBack0;

        // copy back positions of all monomers
        mLog( "Info" ) << "[" << __FILENAME__ << "] copy back monomers from GPU updater to CPU 'molecules' to be used with analyzers\n";
        for( size_t i = 0; i < mIngredients.getMolecules().size(); ++i )
        {
            molecules[i].setAllCoordinates
            (
                mUpdaterGpu.getMonomerPositionInX(i),
                mUpdaterGpu.getMonomerPositionInY(i),
                mUpdaterGpu.getMonomerPositionInZ(i)
            );
        }

        /* update number of total simulation steps already done */
        mIngredients.modifyMolecules().setAge( mIngredients.modifyMolecules().getAge() + mnSteps );

        if ( mLog.isActive( "Stat" ) )
        {
            std::clock_t const t1 = std::clock();
            double const dt = (double) ( t1 - t0 ) / CLOCKS_PER_SEC;    // in seconds
            /* attempted moves per second */
            double const amps = ( (double) mnSteps * mIngredients.getMolecules().size() )/ dt;

            mLog( "Stat" )
            << "[" << __FILENAME__ << "] mcs " << mIngredients.getMolecules().getAge()
            << " with " << amps << " [attempted moves/s]\n"
            << "[" << __FILENAME__ << "] mcs " << mIngredients.getMolecules().getAge()
            << " passed time " << dt << " [s] with " << mnSteps << " MCS\n";
        }

        if ( mLog.isActive( "Benchmark" ) )
        {
            auto const tCopyBack1 = std::chrono::high_resolution_clock::now();
            std::stringstream sBuffered;
            sBuffered << "tCopyback = " << std::chrono::duration<double>( tCopyBack1 - tCopyBack0 ).count() << "s\n";
            mLog( "Benchmark" ) << sBuffered.str();
        }

        return true;
    }

    template< typename T_UCoordinateCuda, bool T_IsPeriodicX, bool T_IsPeriodicY, bool T_IsPeriodicZ >
    inline bool cleanupUpdater()
    {
        UpdaterGPUScBFM_AB_Type< T_UCoordinateCuda, T_IsPeriodicX, T_IsPeriodicY, T_IsPeriodicZ > & mUpdaterGpu = mUpdatersGpu;

        mLog( "Info" ) << "[" << __FILENAME__ << "] cleanup\n";
        mUpdaterGpu.cleanup();

        return true;
    }

    #if defined( USE_UINT8_POSITIONS )
        #define TMP_CHOOSE_CORRECT_TEMPLATED_METHOD_BY_TYPE( NAME, PX, PY, PZ ) \
        ( mCanUseUint8Positions                                                 \
          ? NAME##Updater< uint8_t , PX, PY, PZ >()                             \
          : NAME##Updater< uint16_t, PX, PY, PZ >() )
    #else
        #define TMP_CHOOSE_CORRECT_TEMPLATED_METHOD_BY_TYPE( NAME, PX, PY, PZ ) NAME##Updater< int32_t, PX, PY, PZ >()
    #endif
    #define TMP_CHOOSE_CORRECT_TEMPLATED_METHOD( NAME )     \
    bool result = true;                                     \
    auto const p = ( mIngredients.isPeriodicX() << 2 ) +    \
                   ( mIngredients.isPeriodicY() << 1 ) +    \
                     mIngredients.isPeriodicZ();            \
    switch ( p )                                            \
    {                                                       \
        case 0: result = TMP_CHOOSE_CORRECT_TEMPLATED_METHOD_BY_TYPE( NAME, false, false, false ); break; \
        case 1: result = TMP_CHOOSE_CORRECT_TEMPLATED_METHOD_BY_TYPE( NAME, false, false,  true ); break; \
        case 2: result = TMP_CHOOSE_CORRECT_TEMPLATED_METHOD_BY_TYPE( NAME, false,  true, false ); break; \
        case 3: result = TMP_CHOOSE_CORRECT_TEMPLATED_METHOD_BY_TYPE( NAME, false,  true,  true ); break; \
        case 4: result = TMP_CHOOSE_CORRECT_TEMPLATED_METHOD_BY_TYPE( NAME,  true, false, false ); break; \
        case 5: result = TMP_CHOOSE_CORRECT_TEMPLATED_METHOD_BY_TYPE( NAME,  true, false,  true ); break; \
        case 6: result = TMP_CHOOSE_CORRECT_TEMPLATED_METHOD_BY_TYPE( NAME,  true,  true, false ); break; \
        case 7: result = TMP_CHOOSE_CORRECT_TEMPLATED_METHOD_BY_TYPE( NAME,  true,  true,  true ); break; \
        default: assert( false );                           \
    }

    inline void initialize()
    {
        auto const maxBoxSize = std::max( mIngredients.getBoxX(),
            std::max( mIngredients.getBoxY(), mIngredients.getBoxZ() ) );
        assert( maxBoxSize >= 0 );
        mCanUseUint8Positions = (unsigned int) maxBoxSize <= ( 1llu << ( CHAR_BIT * sizeof( uint8_t ) ) );
        TMP_CHOOSE_CORRECT_TEMPLATED_METHOD( initialize );
        (void) result; /* quelch unused warnings */
    }
    inline bool execute(){ TMP_CHOOSE_CORRECT_TEMPLATED_METHOD( execute ); return result; }
    inline void cleanup(){ TMP_CHOOSE_CORRECT_TEMPLATED_METHOD( cleanup ); (void) result; }

    #undef TMP_CHOOSE_CORRECT_TEMPLATED_METHOD_BY_TYPE
    #undef TMP_CHOOSE_CORRECT_TEMPLATED_METHOD
};
