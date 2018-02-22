
#include <array>
#include <cstdlib>                      // atol
#include <stdint.h>                     // uint8_t, uint32_t (cstdint seems to be C++11)
#include <cstring>
#include <exception>
#include <fstream>                      // getline
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

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
#include "graphColoring.tpp"
#include "SelectiveLogger.hpp"              // __FILENAME__


void printHelp( void )
{
    std::stringstream msg;
    msg << "usage: ./colorBFM [options]\n"
        << "\n"
        << "Simple Simulator for the ScBFM with excluded volume and BondCheck splitted CL-PEG in z on GPU\n"
        << "\n"
        << "    -i, --initial-state <file path>\n"
        << "        (required) specify a BFM file to load the configuration to simulate from\n"
        << "    -s, --save-interval <integer>\n"
        << "        save after every <integer> Monte-Carlo steps to the output file.\n"
        << "    -o, --output <file path>\n"
        << "        all intermediate steps at each save-interval will be appended to this file even if it already exists\n";
    std::cout << msg.str();
}

int main( int argc, char ** argv )
{
    std::string infile; /* BFM file containing positions of monomers and their connections */
    std::string outfile      = "outfile.bfm"; /* at save_interval steps the current state of the simulation will be written to this file */
    uint32_t save_interval   = 0;
    bool bUniformColors      = false;

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
                { "help"         , no_argument      , 0, 'h' },
                { "initial-state", required_argument, 0, 'i' },
                { "output"       , required_argument, 0, 'o' },
                { "save-interval", required_argument, 0, 's' },
                { "uniform"      , required_argument, 0, 'u' },
                { 0, 0, 0, 0 }    // signify end of list
            };
            /* getopt_long stores the option index here. */
            int option_index = 0;
            int c = getopt_long( argc, argv, "hi:o:s:u", long_options, &option_index );

            if ( c == -1 )
                break;

            switch ( c )
            {
                case 'h':
                    printHelp();
                    return 0;
                case 'i':
                    infile = std::string( optarg );
                    break;
                case 'o':
                    outfile = std::string( optarg );
                    break;
                case 's':
                    save_interval = std::atol( optarg );
                    break;
                case 'u':
                    bUniformColors = true;
                    break;
                default:
                    std::cerr << "Unknown option encountered: " << optarg << "\n";
                    return 1;
            }
        }

        std::cerr
        << "infile  = " << infile  << "\n"
        << "outfile = " << outfile << "\n";

        /*************************** read BFM file ***************************/
        /**
         * This is based on the following files doing it in a more roundabout way:
         * updater/UpdaterReadBfmFile.h
         *   Just a TaskManager-usable wrapper for FileImport.h
         *   Important are the constructor, initialize
         * io/FileImport.h
         *   Opens the file for reading and allows easy selection of
         *   conformations stored in it
         *   scanFile: creates a map of file positions for all polymer conformations
         *   Important methods are the constructor, initialize, readHeader,
         *   scanFile, gotoMcs, gotoEnd
         *   Interpretation of assignment strings like !box_x=64 is delegated
         *   to AbstractRead.h
         *   The actual reading of data is done by the read() (for conformation)
         *   and executeRead() methods, which in turn calls registered callbacks
         *   Those callbacks are set by the exportRead method in Ingredients.h,
         *   which in turn calls exportRead for every Feature
         * io/AbstractRead.h
         *   Finds and interpretes strings of the form !key=value
         * feature/FeatureMoleculesIO.h
         *   registers the callback for !number_of_monomers !bonds !add_bonds
         *   and especially !mcs to ReadMcs defined in MoleculesRead.h
         * core/MoleculesRead.h
         *   Finally the file which does the actual reading of the binary data -.-
         *   Important: ReadMcs (execute, processRegularLine), ReadBonds (execute)
         */
        std::ifstream file( infile, std::ios::in | std::ios::binary );

        /* Find and read number of monomers */
        auto findKeyword = []( std::ifstream & file, std::string const & keyword ) -> std::string
        {
            std::string line;
            while ( ! file.eof() && ! file.fail() )
            {
                std::getline( file, line );
                if ( line.compare( 0, keyword.size(), keyword ) == 0 )
                {
                    std::cerr << "Found '" << keyword << "' at postion " << file.tellg() << "\n";
                    return line;
                }
            }
            return "";
        };
        std::string const snMonomers = "!number_of_monomers=";
        file.seekg( 0, std::ios::beg );
        size_t const nMonomers = std::atoi( findKeyword( file, snMonomers ).substr( snMonomers.size() ).c_str() );
        std::cerr << "nMonomers = " << nMonomers << "\n";

        /* allocate data structure for monomers */
        struct Monomer { int x,y,z; std::vector< size_t > neighbors; };
        std::vector< Monomer > monomers( nMonomers );

        /* Read bonds from file (these are only bonds, no positions)
         * Note that this does not include the implied bonds in !mcs !
         * Very confusing choice for this program's needs */
        file.seekg( 0, std::ios::beg );
        findKeyword( file, "!bonds" );
        std::string line;
        size_t a, b = 0;
        while ( ! file.eof() && ! file.fail() &&
                ( std::getline( file, line ), ! line.empty() ) &&
                line[0] != '!' ) // note: look up "c++ comma operator"
        {
            std::stringstream sline( line );
            sline >> a >> b; /* read ascii encoded and space separated bond partners */
            if ( file.fail() )
                throw std::runtime_error( "Error while reading bond partners!" );
            /* add connection (no check for duplicates) */
            if ( a < 20 )
                std::cerr << "Connect " << a-1 << "--" << b-1 << "\n";
            monomers.at( a-1 ).neighbors.push_back( b-1 );
            monomers.at( b-1 ).neighbors.push_back( a-1 );
        }

    #define READ_POSITIONS
    #ifdef READ_POSITIONS
        /* Read bond vector set, needed because the monomer positions are
         * encoded usings these. But not actually needed to extract the
         * connections only. For that it suffices to count the number of bonds
         * per line after !mcs  */
        std::map< int, std::array< int, 3 > > allowedBonds;
        file.seekg( 0, std::ios::beg );
        findKeyword( file, "!set_of_bondvectors" );
        while ( ! file.eof() && ! file.fail() &&
                ( std::getline( file, line ), ! line.empty() ) &&
                line[0] != '!' ) // note: look up "c++ comma operator"
        {
            int dx,dy,dz,iBondVector;
            std::stringstream sline( line );
            sline >> dx >> dy >> dz;
            assert( sline.peek() == ':' );
            sline.ignore(1);
            sline >> iBondVector;
            allowedBonds[ iBondVector ] = { dy, dy, dz };
        }
    #endif

        /* Find last conformation (there might be multiple) */
        std::streampos lastFoundConformation = -1;
        file.seekg( 0, std::ios::beg );
        while ( true )
        {
            line = findKeyword( file, "!mcs" );
            if ( ! line.empty() )
                lastFoundConformation = file.tellg();
            else
                break;
        }
        if ( lastFoundConformation != -1 )
        {
            file.clear();   // reset EOF bit
            file.seekg( lastFoundConformation );
        }
        else
            throw std::invalid_argument( "No conformation (starting with !mcs) found in specified file!" );

        /* Read last conformation / positions into array of connections
         * The format is: !mcs
         *   x y z (coordinates for monomer 0) <binary vector of bond IDs in the
         *   bond set to the next monomer i.e. 1,2,3,...> \n
         *   x y z (coordinates of monomer i which is NOT connected to i-1) <...>
         *    => Data must encompass nMonomers given either by absolute or
         *       relative coordinates */
        size_t iMonomer = 0;
        while ( ! file.eof() && ! file.fail() &&
                ( std::getline( file, line ), ! line.empty() ) &&
                line[0] != '!' ) // note: look up "c++ comma operator"
        {
            std::stringstream sline( line );
            int x,y,z;
            sline >> x >> y >> z; // not stored in binary, but as ascii integers separated by space
            assert( iMonomer < nMonomers );
            if ( ! sline.fail() )
            {
                monomers.at( iMonomer ).x = x;
                monomers.at( iMonomer ).y = y;
                monomers.at( iMonomer ).z = z;
                ++iMonomer;
            }
            else
                throw std::runtime_error( "Couldn't read monomer position!" );
            assert( sline.peek() == ' ' );
            sline.ignore(1); // ignore single space

            /* read the binary coded bond vectors of this linear chain */
            // https://stackoverflow.com/questions/5605125/why-is-iostreameof-inside-a-loop-condition-considered-wrong
            /* note that we are reading the streamstring version of the line, so
             * the end is the end of the line */
            while ( sline.peek() > 0 && ! sline.fail() )
            {
                size_t const iBondVector = sline.get(); // get one Ascii char! This means range limited to [-127,128]!
                #ifdef READ_POSITIONS
                    /* add bond length to current position to get the position
                     * of the connected monomer */
                    auto const & bond = allowedBonds.at( iBondVector );
                    x += bond[0];
                    y += bond[1];
                    z += bond[2];
                    monomers.at( iMonomer ).x = x;
                    monomers.at( iMonomer ).y = y;
                    monomers.at( iMonomer ).z = z;
                #endif
                assert( iMonomer > 0 );
                monomers.at( iMonomer   ).neighbors.push_back( iMonomer-1 );
                monomers.at( iMonomer-1 ).neighbors.push_back( iMonomer   );
                ++iMonomer;
            }
        }

        /* Test output of connections */
        std::cerr << "Connections of monomers:\n";
        for ( size_t iMonomer = 0u; iMonomer < std::min( nMonomers, 20lu ); ++iMonomer )
        {
            std::cerr << "    " << iMonomer << " :";
            for ( auto const & n : monomers[ iMonomer ].neighbors )
                std::cerr << " " << n;
            std::cerr << "\n";
        }

        /* Do the actual coloring */
        std::vector< uint8_t > const & colors = graphColoring<
            std::vector< Monomer > const, uint32_t, uint8_t
        >(
            monomers, nMonomers, bUniformColors,
            []( std::vector< Monomer > const & x, uint32_t const & i ){ return x[i].neighbors.size(); },
            []( std::vector< Monomer > const & x, uint32_t const & i, size_t const & j ){ return x[i].neighbors[j]; }
        );

        /* write out file again, but this time with attribute information we got */
        /* 1.) seek back to beginning of file. Go through file until !attributeSystem found and while doing that copy infile to outfile
         * 2.) If not found, then the loop finished and we just have to append it ...
        */
        {
            std::ofstream coloredFile( outfile, std::ios::out | std::ios::binary );
            std::string line;
            file.clear();
            file.seekg( 0, std::ios::beg );
            bool written = false;
            std::string const sAttributes = "!attributes";
            std::string const sMcs        = "!mcs";
            while ( ! file.eof() && ! file.fail() )
            {
                std::getline( file, line );
                /* if there is no !mcs, then this won't write out attributes,
                 * but in that case the file is invalid anyway */
                if ( ( line.compare( 0, sAttributes.size(), sAttributes ) == 0 ||
                       line.compare( 0, sMcs       .size(), sMcs        ) == 0 ) &&
                     ! written )
                {
                    written = true;
                    std::cerr << "Writing out " << colors.size() << " colors\n";
                    coloredFile << sAttributes <<"\n";
                    for ( size_t i = 0u; i < colors.size(); ++i )
                        coloredFile << i+1 << "-" << i+1 << ":" << (int) colors[i] << "\n";
                    coloredFile << "\n";
                }

                if ( line.compare( 0, sAttributes.size(), sAttributes ) == 0 )
                {
                    std::cerr << "Found '" << sAttributes << "' at postion " << file.tellg() << "\n";
                    /* skip all old attributes before writing out */
                    while ( ! file.eof() && ! file.fail() &&
                            ( std::getline( file, line ), ! line.empty() ) &&
                            ! ( '0' <= line[0] && line[0] <= '9' ) );
                }
                coloredFile << line << "\n";
            }
            std::cerr << "Written color attributes to '" << outfile << "'\n";
            // destructor of coloredFile closes it
        }
    }
    catch( std::exception const & e )
    {
        std::cerr << "[" << __FILENAME__ << "] Caught exception: " << e.what() << std::endl;;
    }
    return 0;

}

/**
 * make && ./colorBFM -i ../tests/coloring-test-Hydrogel_HEP_3__PEG_3_NStar_117__NoPerXYZ64_Connected.bfm -o colored.bfm
 *     Maximum number of neighbors per monomer: 5
 *     Number of isolated subgraphs / polymers: 1
 *     Number of colors needed for the polymer system: 4
 *     Color usage frequencies: A: 279x (44.9275%), B: 267x (42.9952%), C: 65x (10.467%), D: 10x (1.61031%)
 *
 * make && ./SimulatorCUDAGPUScBFM_AB_Type -i ../tests/coloring-test-Hydrogel_HEP_3__PEG_3_NStar_117__NoPerXYZ64_Connected.bfm -m 1000 -s 1000 -o ./result.bfm -e ../tests/resultNormPscBFM.seeds -g 1
 *     Maximum number of neighbors per monomer: 5
 *     Number of isolated subgraphs / polymers: 1
 *     Number of colors needed for the polymer system: 4
 *     Color usage frequencies: A: 279x (44.9275%), B: 267x (42.9952%), C: 65x (10.467%), D: 10x (1.61031%)
 *
 * => identical :) -> works
 *
 * ../../Programs/LeMonADE-Viewer/build/LeMonADE-Viewer colored.bfm
 *   !setColorAttributes:0=(228,26,28)
 *   !setColorAttributes:1=(55,126,184)
 *   !setColorAttributes:2=(77,175,74)
 *   !setColorAttributes:3=(152,78,163)
 * => LeMonADE-Viewer expects [0,1) -.- but does not tell so
 * => if !attributeSystem comes after !mcs it seems to be ignored!
 * => if !attributeSystem comes before, then LeMonADE-Viewer segfaults Oo?
 *      => seems to be because of autodelete of trailing spaces >:O
 */
