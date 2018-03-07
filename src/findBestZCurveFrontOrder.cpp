/*
g++ -std=c++11 -Wall -Wextra findBestZCurveFrontOrder.cpp && ./a.out
*/

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

unsigned int part1by2( unsigned int n )
{
    n &= 0x000003ff;
    n = (n ^ (n << 16)) & 0xff0000ff; // 0b 0000 0000 1111 1111
    n = (n ^ (n <<  8)) & 0x0300f00f; // 0b 1111 0000 0000 1111
    n = (n ^ (n <<  4)) & 0x030c30c3; // 0b 0011 0000 1100 0011
    n = (n ^ (n <<  2)) & 0x09249249; // 0b 1001 0010 0100 1001
    return n;
}

uint32_t unpart1by2( uint32_t n )
{
    n &= 0x09249249;
    n = (n ^ (n >>  2)) & 0x030c30c3;
    n = (n ^ (n >>  4)) & 0x0300f00f;
    n = (n ^ (n >>  8)) & 0xff0000ff;
    n = (n ^ (n >> 16)) & 0x000003ff;
    return n;
}

long unsigned int linearizeALongZCurve( int const x, int const y, int const z )
{
    return
     ( part1by2( x )      ) |
     ( part1by2( y ) << 1 ) |
     ( part1by2( z ) << 2 );
}

void unlinearizeAlongZCurve( long unsigned int i, int * const x, int * const y, int * const z )
{
    *x = unpart1by2( i      );
    *y = unpart1by2( i >> 1 );
    *z = unpart1by2( i >> 2 );
}

/**
 * @param[out] axis axis along which it moves 0:x, 1:y, 2:z
 * @param[out] sign whether the front is in + or - xyz direction
 * @param[out] da,db relative coordinate system which is 2D. a and b are those
 *                   coordinates  which are actually changing in the front and
 *                   in order of the original, i.e. either yz, xz or xy
 *                   depending whether the axis is x, y or z
 */
void toFrontCoordinates
(
    int const dx, int const dy, int const dz,
    int * const axis,
    int * const sign,
    int * const da,
    int * const db
)
{
    int dr[3] = { dx, dy, dz };
    int reduced[2];
    int iReduced = 0;
    for ( auto i = 0u; i < 3u; ++i )
    {
        if ( std::abs( dr[i] ) == 2 )
        {
            *axis = i;
            *sign = dr[i] / std::abs( dr[i] );
        }
        else
        {
            assert( iReduced < 2 );
            reduced[ iReduced++ ] = dr[i];
        }
    }
    *da = reduced[0];
    *db = reduced[1];
}

int main( void )
{
    auto const nLargeBox    = 8;
    auto const nDiffToCheck = 2;

    auto const nTotalFrontCells = 3 /* nDims */ * 2 /* nSigns */ * 9 /* frontSize**(nDims-1) */;
    std::vector< std::vector< int > > drFound( nTotalFrontCells, std::vector< int >( 3 ) );

    for ( int cx = 2; cx < nLargeBox - 2; ++cx )
    for ( int cy = 2; cy < nLargeBox - 2; ++cy )
    for ( int cz = 2; cz < nLargeBox - 2; ++cz )
    {
        std::cout << "Best whole ordering for case cx=(" << cx << "," << cy << "," << cz << "): ";
        /*
        for ( int ix = 0; ix < nLargeBox; ++ix )
        for ( int iy = 0; ix < nLargeBox; ++iy )
        for ( int iz = 0; ix < nLargeBox; ++iz )
        */
        int nFrontCellsFound = 0;
        /* go through the lattice as it is stored in the memory */
        for ( auto i = 0u; i < std::pow( nLargeBox, 3 ); ++i )
        {
            int ix,iy,iz;
            unlinearizeAlongZCurve( i, &ix, &iy, &iz );
            /* check if we hit one of the fronts we have to check, this is
             * the same as asking if it is on INSIDE cells of the outer shell
             * i.e. check if on shell and then chack if inside certain
             * quadratic tubes along all 3 axes ... ... */
            if ( ( std::abs( ix - cx ) == 2 ) ||
                 ( std::abs( iy - cy ) == 2 ) ||
                 ( std::abs( iz - cz ) == 2 ) )
            {
                /* count number of dimensions where we are not inside a cube */
                if ( ( ( std::abs( ix - cx ) < 2 ) +
                       ( std::abs( iy - cy ) < 2 ) +
                       ( std::abs( iz - cz ) < 2 ) ) == 2 )
                {
                    assert( nFrontCellsFound < nTotalFrontCells );
                    drFound.at( nFrontCellsFound ) = { ix - cx, iy - cy, iz - cz };
                    ++nFrontCellsFound;
                    std::cout << "(" << ix - cx << "," << iy - cy << "," << iz - cz << "), ";
                }
            }
        }
        std::cout << std::endl;
        assert( nFrontCellsFound == nTotalFrontCells );

        std::vector< std::string > sAxes = { "x", "y", "z" };
        for ( int axis = 0; axis < sAxes.size(); ++axis )
        for ( auto sign : { -1, 1 } )
        {
            int axis2, sign2, da, db;
            std::cout << ( sign == 1 ? "+" : "-" ) << sAxes.at( axis ) << ": ";
            for ( auto i = 0u; i < drFound.size(); ++i )
            {
                toFrontCoordinates( drFound[i][0], drFound[i][1], drFound[i][2],
                    &axis2, &sign2, &da, &db
                );
                if ( axis2 == axis && sign == sign2 )
                {
                    std::cout << "(" << da << "," << db << "), ";
                }
            }
            std::cout << std::endl;
        }
    }

    return 0;
}
