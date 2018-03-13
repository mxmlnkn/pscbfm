
#include <chrono>                       // std::chrono::high_resolution_clock
#include <cstring>
//#include <cstdint>                      // uint32_t (C++11)
#include <iostream>
#include <sstream>

#include <getopt.h>

#include <LeMonADE/utility/RandomNumberGenerators.h>
#include <LeMonADE/core/ConfigureSystem.h>
#include <LeMonADE/core/Ingredients.h>
#include <LeMonADE/feature/FeatureMoleculesIO.h>
#include <LeMonADE/feature/FeatureAttributes.h>
#include <LeMonADE/feature/FeatureExcludedVolumeSc.h>
#include <LeMonADE/feature/FeatureFixedMonomers.h>
#include <LeMonADE/utility/TaskManager.h>
#include <LeMonADE/updater/UpdaterReadBfmFile.h>
#include <LeMonADE/updater/UpdaterSimpleSimulator.h>

#include "GPUScBFM_AB_Type.h"
#include "SelectiveLogger.hpp"              // __FILENAME__


void printHelp( void )
{
    std::stringstream msg;
    msg << "usage: ./SimulatorCUDAGPUScBFM_AB_Type [options]\n"
        << "\n"
        << "Simple Simulator for the ScBFM with excluded volume and BondCheck splitted CL-PEG in z on GPU\n"
        << "\n"
        << "    -e, --seeds <file path>\n"
        << "        specify a seed file to use for reproducible result simulations. Currently this should contain 256+2 random 32-Bit values (or more).\n"
        << "    -i, --initial-state <file path>\n"
        << "        (required) specify a BFM file to load the configuration to simulate from\n"
        << "    -m, --max-mcs <integer>\n"
        << "        (required) specifies the total Monte-Carlo steps to simulate.\n"
        << "    -s, --save-interval <integer>\n"
        << "        save after every <integer> Monte-Carlo steps to the output file.\n"
        << "    -o, --output <file path>\n"
        << "        all intermediate steps at each save-interval will be appended to this file even if it already exists\n"
        << "    -g, --gpu <integer>\n"
        << "        specify the GPU to use. The ID goes from 0 to the number of GPUs installed - 1\n"
        << "    -v, --version\n"
        ;
    std::cout << msg.str();
}

int main( int argc, char ** argv )
{
    using hrclock = std::chrono::high_resolution_clock;
    auto const tProgram0 = hrclock::now();

    std::string infile; /* BFM file containing positions of monomers and their connections */
    std::string outfile      = "outfile.bfm"; /* at save_interval steps the current state of the simulation will be written to this file */
    uint32_t max_mcs         = 0; /* how many Monte-Carlo steps to simulate */
    uint32_t save_interval   = 0;
    int      iGpuToUse       = 0;
    std::string seedFileName = "";

    try
    {

        if ( argc <= 1 )
        {
            printHelp();
            return 0;
        }

        /* Parse command line arguments */
        while ( true )
        {
            static struct option long_options[] = {
                { "seeds"        , required_argument, 0, 'e' },
                { "gpu"          , required_argument, 0, 'g' },
                { "help"         , no_argument      , 0, 'h' },
                { "initial-state", required_argument, 0, 'i' },
                { "max-mcs"      , required_argument, 0, 'm' },
                { "output"       , required_argument, 0, 'o' },
                { "save-interval", required_argument, 0, 's' },
                { "version"      , no_argument      , 0, 'v' },
                { 0, 0, 0, 0 }    // signal end of list
            };
            /* getopt_long stores the option index here. */
            int option_index = 0;
            int c = getopt_long( argc, argv, "e:g:hi:m:o:s:v", long_options, &option_index );

            if ( c == -1 )
                break;

            switch ( c )
            {
                case 'e':
                    seedFileName = std::string( optarg );
                    break;
                case 'h':
                    printHelp();
                    return 0;
                case 'g':
                    iGpuToUse = std::atoi( optarg );
                    break;
                case 'i':
                    infile = std::string( optarg );
                    break;
                case 'm':
                    max_mcs = std::atol( optarg );
                    break;
                case 'o':
                    outfile = std::string( optarg );
                    break;
                case 's':
                    save_interval = std::atol( optarg );
                    break;
                case 'v':
                    std::cout
                        << "Version compiled on " << __DATE__
                        << " using commit " << GIT_COMMIT_HASH << "-" << GIT_DIRTY
                        << " on branch " << GIT_BRANCH << std::endl;
                    break;
                default:
                    std::cerr << "Unknown option encountered: " << optarg << "\n";
                    return 1;
            }
        }

        std::cerr
        << "infile          = " << infile        << "\n"
        << "outfile         = " << outfile       << "\n"
        << "max_mcs         = " << max_mcs       << "\n"
        << "save_interval   = " << save_interval << "\n";

        //seed the globally available random number generators
        RandomNumberGenerators rng;
        if ( ! seedFileName.empty() )
        {
            std::cerr << "Use seeds from: " << seedFileName << "\n";
            rng.seedAll( seedFileName );
        }
        else
            rng.seedAll();

        /* Check the initial values. Note that the drawing of these random
         * values can't be omitted, or else all subsequent random numbers
         * will shift / change! */
        std::cerr << "std rand: " << std::setw(12) << std::rand()       << " =?= 764080779"  << "\n";
        std::cerr << "RNG rand: " << std::setw(12) << rng.r250_rand32() << " =?= 4223731124" << "\n";
        std::cerr << "RNG rand: " << std::setw(12) << rng.r250_drand()  << " =?= 0.803876"   << "\n";

        /*
        FeatureExcludedVolume<> is equivalent to FeatureExcludedVolume< FeatureLattice< bool > >
        typedef LOKI_TYPELIST_3( FeatureBondset, FeatureAttributes,
            FeatureLattice< uint8_t > FeatureExcludedVolume< FeatureLatticePowerOfTwo<> > )
            Features;
        */
        typedef LOKI_TYPELIST_3( FeatureMoleculesIO, FeatureAttributes,
                                 FeatureExcludedVolumeSc<> ) Features;

        typedef ConfigureSystem< VectorInt3, Features, 8 > Config;
        typedef Ingredients< Config > Ing;
        Ing myIngredients;

        TaskManager taskmanager;
        taskmanager.addUpdater( new UpdaterReadBfmFile<Ing>( infile, myIngredients,UpdaterReadBfmFile<Ing>::READ_LAST_CONFIG_SAVE ), 0 );
        //here you can choose to use MoveLocalBcc instead. Careful though: no real tests made yet
        //(other than for latticeOccupation, valid bonds, frozen monomers...)
        taskmanager.addUpdater( new GPUScBFM_AB_Type<Ing>( myIngredients, save_interval, iGpuToUse ) );

        taskmanager.addAnalyzer( new AnalyzerWriteBfmFile<Ing>( outfile, myIngredients ) );

        taskmanager.initialize();

        auto const tTasks0 = hrclock::now();
        taskmanager.run( max_mcs / save_interval );

        auto const tTasks1 = hrclock::now();
        std::stringstream sBuffered;
        sBuffered << "tTaskLoop = " << std::chrono::duration<double>( tTasks1 - tTasks0 ).count() << "s\n";
        std::cerr << sBuffered.str();

        taskmanager.cleanup();
    }
    catch( std::exception const & e )
    {
        std::cerr << "[" << __FILENAME__ << "] Caught exception: " << e.what() << std::endl;;
    }

    auto const tProgram1 = hrclock::now();
    std::stringstream sBuffered;
    sBuffered << "tProgram = " << std::chrono::duration<double>( tProgram1 - tProgram0 ).count() << "s\n";
    std::cerr << sBuffered.str();
    return 0;
}
