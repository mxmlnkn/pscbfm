
#pragma once

#include <cmath>
#include <cstdint>                  // UINT64_MAX
#include <map>
#include <stack>
#include <stdexcept>
#include <string>
#include <utility>                  // std::pair
#include <vector>

#include <LeMonADE/utility/Vector3D.h>

#define DEBUG_ANALYZERMSD 10


class OnlineStatistics
{
private:
    std::vector< double > mMoments; // 0: nValues, 1: mean * nValues, ... note: these are not central moments like the variance is!
	double mValue;
	double mMin, mMax;

    /**
     * Returns the binomial coefficient, i.e.,
     * n!/(k!(n-k)!) = \sum_{i=1}^k (n-k+i)/i = \sum_{i=1}^k (n-(i-1))/i
     */
    double choose( unsigned long int nTotal, unsigned long int nToChoose )
    {
        /* performance optimization using the identity k!(n-k)! = (n-k)!(n-(n-k))! */
        if ( nToChoose > nTotal / 2 )
            nToChoose = nTotal - nToChoose;
        double result = 1;
        /**
         * Storing the moments as divided by the total number of values,
         * should lead to less precision because of the frequent error-
         * accumulating n/(n+1) multiplication.
         * The effect on the extended range is also limited, as e.g. for
         * 1e10 values this way only saves 10 exponents range out of a
         * max double exponent of 308. So it is negligible.
         */
        for ( unsigned long int i = 0u; i < nToChoose; ++i )
            result *= double( nTotal-i ) / double( i+1 );
        return result;
    }

public:
	OnlineStatistics( unsigned int nMaxMoment = 2 ) : mMoments( 1+nMaxMoment, 0 ){}

    inline void addValue( double const x )
    {
        mValue = x;
        double pow = 1;
        for ( size_t i = 0u; i < mMoments.size(); ++i, pow *= x )
            mMoments[i] += pow;
        if ( mMoments[0] == 1 )
			mMin = mMax = x;
	}

    inline double getMoment( unsigned int i )
    {
        if ( i >= mMoments.size() )
            throw std::invalid_argument( "Requested moment was not configured to be calculated by this class." );
        return mMoments.at(i) / mMoments.at(0);
    }

    /**
     * @see https://en.wikipedia.org/wiki/Moment_(mathematics)#Central_moments_in_metric_spaces
     * Central moment: <(x-<x>)^i> = \sum_{j=0}^i choose(i,j) <x^{j-i}> (-1)^j <x>^j
     */
    inline double getCentralMoment( unsigned int iMoment )
    {
        if ( iMoment >= mMoments.size() )
            throw std::invalid_argument( "Requested moment was not configured to be calculated by this class." );
        double cmom = 0;
        for ( unsigned int i = 0; i <= iMoment; ++i )
        {
            cmom += choose( mMoments.size(), i ) * getMoment( iMoment-i ) *
                    ( i % 2 == 0 ? +1 : -1 ) * std::pow( getMoment(1), i );
        }
        /* for the central moment we have to use the mean which reduces the
         * degrees of freedom by 1, therefore return with Bessel's correction */
        return cmom * ( mMoments.at(0) / ( mMoments.at(0)-1 ) );
    }

    /* some convenience named moments */
    inline double getMean     (){ return getMoment(1); }
    inline double getVariance (){ return getCentralMoment(2); }
    inline double getStddev   (){ return std::sqrt( getCentralMoment(2) ); }
	inline double getNValues  (){ return mMoments.at(0); }
	inline double getMin      (){ return mMin; }
	inline double getMax      (){ return mMax; }
    inline double getLastValue(){ return mValue; }
};



template< class T_Ingredients >
class AnalyzerMsd : public AbstractAnalyzer
{
private:
    T_Ingredients const & mIngredients;

    uint64_t mnMcsToSkip; /* number of simulation steps to skip before beginning statistics */
    uint64_t mDeltaTRealization; /* number of simulation steps before starting the next realization. Should be high enough that the resulting subsequences are largely uncorrelated */
    uint64_t miMcsLastRealization; /* MCS in which we last began a new ensemble subsequence */
    uint64_t mDeltaTEvaluate; /* the smaller the more data points will be dumped to file, the larger the file and the more fine-grained the resulting plot. */
    uint64_t miMcsLastEvaluation; /* the last time we actually evaluated a new data point. This is in order to work with this analyzer being called more often than we need plot points for g_i, i.e. DeltaTPlot a.k.a DeltaTEvaluate */

    uint64_t mnChains;
    std::vector< uint64_t > mvnMonomersPerChain;
    std::vector< std::vector< uint32_t > > mviEndMiddleMonomers; // for each chain this contains in this order (iStart,iMiddle,iEnd)

    static auto constexpr mnMSDs = 5; // MSDs
    std::vector< std::vector< OnlineStatistics > > mMSDs; // means square displacements we are interested in. mMSDs[ iEvalPoint ][ iAverage ], the outer could also be a list as we need to often resize it :/
    std::vector< uint64_t > mEvalTimes; /* which index of mMSDs does correspond to which time difference in simulation time steps. For the easiest case it should just be index * deltaTEvaluate, but not so after the subsequence halfing has been implemented */
    std::vector< std::vector< std::vector< VectorDouble3 > > > mvFirstValues; // mvFirstValues[ iSubsequence ][ iAverage ][ iChain ][ iCoord ] these are needed to calculate the MSDs, i.e. <( CurrentValues(t) - mvFirstValues )^2>
    std::vector< VectorDouble3 > mvFirstValuesBoxCOM;
    std::vector< uint64_t > mviFirstMcs; /* first simulation step for the given subsequence */

    std::string msFilename;

    void getLinearPolymerSystemInfo( void );

public:

    /**
     * @param[in] rnMcsToSkip the numbre of simulation time steps to skip.
     *            This is necessary to let the system reach equilibrium before
     *            we can start analyzing it.
     * @param[in] nMaxSubsequences the maximum of time-shifted subsequences.
     *            When this maximum is reached, the number of sequences will
     *            be halved and deltaTRealization will be doubled in order to
     *            get more widely spaced subsequences!
     *            The larger rDeltaTRealization is, the larger nMaxSubsequences
     *            can be set for still uncorrelated subsequences. Just beware
     *            that for tEvaluation-nMcsToSkip > deltaTRealization the
     *            subsequences will become correlated!
     */
    inline AnalyzerMsd
    (
        T_Ingredients          const & rIngredients      ,
        unsigned long long int const   rnMcsToSkip       ,
        unsigned long long int const   rDeltaTRealization,
        unsigned long long int const   rDeltaTEvaluate   ,
        unsigned long long int const   rnMaxSubsequences ,
        std::string            const & rFilename
    )
    : mIngredients       ( rIngredients       ),
      mnChains           ( 0                  ),
      mnMcsToSkip        ( rnMcsToSkip        ),
      mDeltaTRealization ( rDeltaTRealization ),
      mDeltaTEvaluate    ( rDeltaTEvaluate    ), /* in the future make this an argument and test that it works */
      miMcsLastEvaluation( -mDeltaTEvaluate   ), /* ensure that MCS = 0 can already be evaluated */
      msFilename         ( rFilename          )
    {
        mvFirstValues      .reserve( rnMaxSubsequences );
        mvFirstValuesBoxCOM.reserve( rnMaxSubsequences );
        mviFirstMcs        .reserve( rnMaxSubsequences );
    }

    /* give a hint about the number of data points we will have in the end */
    inline void reserve( unsigned long long int rnMcsMax )
    {
        auto const nEvalPoints = ( rnMcsMax + mDeltaTEvaluate - 1 )/ mDeltaTEvaluate;
        mMSDs     .reserve( nEvalPoints );
        mEvalTimes.reserve( nEvalPoints );
    }

    inline virtual ~AnalyzerMsd(){};

    inline virtual void initialize()
    {
        getLinearPolymerSystemInfo();
        execute();
    }
    virtual bool execute();
    virtual void cleanup();
};

/**
 * Counts the number of chains and determines the chain lengths
 */
template< class T_Ingredients >
inline void AnalyzerMsd< T_Ingredients >::getLinearPolymerSystemInfo( void )
{
    auto const & molecules = mIngredients.getMolecules();
    auto const nMonomers = molecules.size();

    /* pick arbitrary monomer and count the length of the chain it is in
     * also find the two end-monomers and using that also store and
     * find the middle monomer
     * check whether it really is a polymer system consisting only of
     * linear chains. And also count the number of chains */
    std::vector< bool > visited( nMonomers, false );
    for ( auto iMonomer = 0; iMonomer < nMonomers; ++iMonomer )
    {
        if ( visited[ iMonomer ] )
            continue;
        ++mnChains;

        std::vector< uint32_t > viEndMonomers( 0 );
        std::stack< uint64_t > vToDo;

        uint64_t nMonomersPerChain = 1;
        vToDo.push( iMonomer );
        visited[ iMonomer ] = true;

        while ( ! vToDo.empty() )
        {
            auto const iChainMonomer = vToDo.top();
            vToDo.pop();

            auto const nNeighbors = molecules.getNumLinks( iChainMonomer );
            if ( nNeighbors > 2 ) /* 0 neighbors may exist for "chains" with only 1 monomer */
                throw std::invalid_argument( "Some monomers have more than two bonds, indicating that this polymer system does not only consist of linear chains. This Analyzer module only works with linear chains!" );

            if ( nNeighbors <= 1 )
                viEndMonomers.push_back( iChainMonomer );

            /* Push unvisited monomer to the todo list */
            for ( auto iBond = 0u; iBond < nNeighbors; ++iBond )
            {
                auto const iNeighbor = molecules.getNeighborIdx( iChainMonomer, iBond );
                if ( ! visited[ iNeighbor ] )
                {
                    ++nMonomersPerChain;
                    visited[ iNeighbor ] = true;
                    vToDo.push( iNeighbor );
                }
            } // loop over neighbors per chain monomer
        } // loop over monomers inside a chain

        /* "chains" with only one monomer have the same start and end */
        if ( nMonomersPerChain == 1 && viEndMonomers.size() == 1 )
            viEndMonomers.push_back( viEndMonomers[0] );

        if ( viEndMonomers.size() != 2 )
            throw std::invalid_argument( "Found not exactly 2 end monomers! The polymer system might be weird, e.g. end monomers with have loops or ring polymers." );

        /**
         * Find middle monomer by following the neighbors of an end
         * monomer floor( nMonomersPerChain / 2 ) times
         * round down so that this works with only one monomer per chain, too
         */
        uint32_t iMiddleMonomer = viEndMonomers.at(0);
        uint32_t iPrevMonomer   = iMiddleMonomer;
        for ( auto i = 0u; i < nMonomersPerChain / 2; ++i )
        {
            if ( molecules.getNeighborIdx( iMiddleMonomer, 0 ) != iPrevMonomer )
            {
                iPrevMonomer   = iMiddleMonomer;
                iMiddleMonomer = molecules.getNeighborIdx( iMiddleMonomer, 0 );
            }
            else
            {
                iPrevMonomer   = iMiddleMonomer;
                iMiddleMonomer = molecules.getNeighborIdx( iMiddleMonomer, 1 );
            }

            /* with all the above checks, this should never happen, at least not for linear chains */
            if ( molecules.getNumLinks( iMiddleMonomer ) < 2 )
                throw std::runtime_error( "Unexpected end monomer encountered when iterating over the polymer to find the middle monomer." );
        }

        /* store results */
        mvnMonomersPerChain.push_back( nMonomersPerChain );
        mviEndMiddleMonomers.push_back( { viEndMonomers.at(0), iMiddleMonomer, viEndMonomers.at(1) } );
    } // loop over all monomers

    #if DEBUG_ANALYZERMSD >= 10
    {
        std::cout << "Found " << mnChains << " linear polymers with ";
        uint64_t min = UINT64_MAX, max = 0;
        for ( auto const & x : mvnMonomersPerChain )
        {
            min = std::min( min, x );
            max = std::max( max, x );
        }
        if ( min == max )
            std::cout << min;
        else
            std::cout << "at least " << min << " at at most " << max;
        std::cout << " monomers per chain" << std::endl;
    }
    #endif
}

/**
 * Calculate the mean square displacements to all time starting points for
 * each realization subsequence and appen to statistics.
 * @verbatim
 * iCall -------------->
 * +---------------- 1st subsequence
 *     +------------ 2nd
 *         +-------- 3rd
 * ^   ^   ^
 * +-tStart+
 * <---> mnCallsToNextRealization
 * @endverbatim
 */
template< class T_Ingredients >
inline bool AnalyzerMsd< T_Ingredients >::execute()
{
    auto const & molecules = mIngredients.getMolecules();

    if ( molecules.getAge() % 100000 == 0 )
        std::cout << "nMonomers = " << mIngredients.getMolecules().size() << " simulation is at step " << molecules.getAge() << std::endl;

    /* append the mean-square displacements to the ensemble averages */
    if ( molecules.getAge() - miMcsLastEvaluation < mDeltaTEvaluate )
        return true;
    miMcsLastEvaluation = molecules.getAge();

    /* Calculate box and chain COM. The other MSD values need no complex calculation */
    std::vector< VectorDouble3 > vChainCOMs( mnChains, VectorDouble3( 0,0,0 ) );
    //#pragma omp parallel for
    for ( uint32_t iChain = 0; iChain < mnChains; ++iChain )
    {
        /* iterate over chain monomers -> would make a nice iterator ... */
        uint32_t iChainMonomer = mviEndMiddleMonomers.at( iChain )[0];
        uint32_t iPrevMonomer = iChainMonomer;
        uint64_t nMonomersInChain = 0;
        while ( true )
        {
            vChainCOMs[ iChain ] += VectorDouble3( molecules[ iChainMonomer ] );
            ++nMonomersInChain;

            /* go to next neighbor */
            if ( molecules.getNeighborIdx( iChainMonomer, 0 ) != iPrevMonomer )
            {
                iPrevMonomer  = iChainMonomer;
                iChainMonomer = molecules.getNeighborIdx( iChainMonomer, 0 );
            }
            else if ( molecules.getNumLinks( iChainMonomer ) >= 2 )
            {
                iPrevMonomer  = iChainMonomer;
                iChainMonomer = molecules.getNeighborIdx( iChainMonomer, 1 );
            }
            else break;
        }
        vChainCOMs[iChain] /= (double) nMonomersInChain;
    }

    VectorDouble3 boxCOM( 0,0,0 );
    for ( uint32_t iChain = 0; iChain < mnChains; ++iChain )
        boxCOM += vChainCOMs[ iChain ];
    boxCOM /= (double) mnChains;

    /* if enough time has passed since the last creation, then
     * create new subsequence, i.e. add new initial values for them  */
    auto const nSubsequences = mvFirstValues.size();
    if ( molecules.getAge() - miMcsLastRealization >= mDeltaTRealization )
    {
        miMcsLastRealization = molecules.getAge();

        mvFirstValues.push_back( std::vector< std::vector< VectorDouble3 > >( mnMSDs,
            std::vector< VectorDouble3 >( mnChains, VectorDouble3(0,0,0) ) ) );
        for ( uint32_t iChain = 0; iChain < mnChains; ++iChain )
        {
            mvFirstValues.back()[0][iChain] = (VectorDouble3) molecules[ mviEndMiddleMonomers[iChain][0] ];
            mvFirstValues.back()[1][iChain] = (VectorDouble3) molecules[ mviEndMiddleMonomers[iChain][1] ];
            mvFirstValues.back()[2][iChain] = (VectorDouble3) molecules[ mviEndMiddleMonomers[iChain][2] ];
            mvFirstValues.back()[3][iChain] = vChainCOMs[iChain];
            mvFirstValues.back()[4][iChain] = mvFirstValues.back()[1][iChain] - vChainCOMs[iChain];
        }
        mvFirstValuesBoxCOM.push_back( boxCOM );
        mviFirstMcs.push_back( molecules.getAge() );
    }

    /* Allocate enough space and save the time */
    for ( size_t iSubsequence = 0u; iSubsequence < nSubsequences; ++iSubsequence )
    {
        auto const iEval = ( molecules.getAge() - mviFirstMcs.at( iSubsequence ) ) / mDeltaTEvaluate;
        if ( iEval >= mMSDs.size() )
        {
            /* can't use resize, as it might invalidate the data! */
            for ( size_t i = 0; iEval >= mMSDs.size(); ++i )
                mMSDs.push_back( std::vector< OnlineStatistics >( mnMSDs, OnlineStatistics() ) );
            for ( size_t i = 0; iEval >= mEvalTimes.size(); ++i )
                mEvalTimes.push_back( 0 );
            mEvalTimes[ iEval ] = molecules.getAge() - mviFirstMcs.at( iSubsequence );
        }
    }

    /* calculate MSDs for all non-new subsequences */
    // can't parallelize this, because the addValue call is not thread-safe!
    for ( size_t iSubsequence = 0u; iSubsequence < nSubsequences; ++iSubsequence )
    {
        auto const iEval = ( molecules.getAge() - mviFirstMcs.at( iSubsequence ) ) / mDeltaTEvaluate;
        /* if there are more than one chain, then we can subtract the change of the total center of mass, i.e., subtract the global fluctuations from the polymer fluctuations. For one chain the global fluctuation is equal to the polymer fluctuation and therefore would make everything we are interested in zero */
        auto const diffBoxCOM = boxCOM - mvFirstValuesBoxCOM.at( iSubsequence );

        /* this is basically already the ensemble averaging over each available chain */
        //# pragma omp parallel for
        for ( uint32_t iChain = 0; iChain < mnChains; ++iChain )
        {
            /* MSDs of first, middle and end monomer */
            for ( auto j = 0u; j < mviEndMiddleMonomers[iChain].size(); ++j )
            {
                auto diff = (VectorDouble3) molecules[ mviEndMiddleMonomers[iChain][j] ]
                          - mvFirstValues.at( iSubsequence )[j][iChain];
                if ( mnChains > 1 ) diff -= diffBoxCOM;
                mMSDs.at( iEval ).at( j ).addValue( diff * diff );
            }
            /* MSD for chain COM */
            {
                auto diff = vChainCOMs[iChain] - mvFirstValues.at( iSubsequence )[3][iChain];
                if ( mnChains > 1 ) diff -= diffBoxCOM;
                mMSDs.at( iEval ).at( 3 ).addValue( diff * diff );
            }
            /* MSD for distance between middle monomer and COM */
            {
                auto diff = ( (VectorDouble3) molecules[ mviEndMiddleMonomers[iChain][1] ]
                          - vChainCOMs[iChain] ) - mvFirstValues.at( iSubsequence )[4][iChain];
                mMSDs.at( iEval ).at( 4 ).addValue( diff * diff );
            }
        }
    }
}

template< class T_Ingredients >
inline void AnalyzerMsd< T_Ingredients >::cleanup()
{
    std::cerr << "[AnalyzerMsd::cleanup]\n";
    std::ofstream file( msFilename, std::ios::out );

    {
        uint64_t min = UINT64_MAX, max = 0;
        for ( auto const & x : mvnMonomersPerChain )
        {
            min = std::min( min, x );
            max = std::max( max, x );
        }
        auto const density = 8. * mIngredients.getMolecules().size() / (
            mIngredients.getBoxX() *
            mIngredients.getBoxY() *
            mIngredients.getBoxZ() );

        file
        << "# nMinMonomersPerChain = " << min                  << "\n"
        << "# nMaxMonomersPerChain = " << max                  << "\n"
        << "# nChains = "              << mnChains             << "\n"
        << "# numberDensity = "        << density              << "\n"
        << "# nMcsSkipped = "          << mnMcsToSkip          << "\n"
        << "# deltaTRealization = "    << mDeltaTRealization   << "\n"
        << "# iMcsLastRealization = "  << miMcsLastRealization << "\n"
        << "# deltaTEvaluate = "       << mDeltaTEvaluate      << "\n"
        << "# iMcsLastEvaluation = "   << miMcsLastEvaluation  << "\n"
        << "# nSubsequences = "        << mvFirstValues.size() << "\n"
        << "# g1 to g5 are in this order the MSDs of: first monomer, middle monomer, end monomer, chain center of mass (COM), middle monomer - chain COM\n"
        << "# dt nSamples g1 sg1 g2 sg2 g3 sg3 g4 sg4 g5 sg5\n";
    }

    for ( size_t iEval = 0; iEval < mMSDs.size(); ++iEval )
    {
        file << mEvalTimes.at( iEval ) << " " << mMSDs.at( iEval ).at(0).getNValues();
        /* return means and standard deviations on the means which is
         * standard deviation / sqrt(n) ! */
        for ( auto iAverage = 0u; iAverage < mnMSDs; ++iAverage )
        {
            file << " " << mMSDs.at( iEval ).at( iAverage ).getMean()
                 << " " << mMSDs.at( iEval ).at( iAverage ).getStddev();
        }
        file << "\n";
    }
}
