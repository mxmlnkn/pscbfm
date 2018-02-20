#pragma once

#ifndef INLINE_GRAPHCOLORING
#   define INLINE_GRAPHCOLORING inline
#endif

#include <cstdint>                      // uint8_t
#include <functional>


template< class T_Neighbors, typename T_Id = size_t, typename T_Color = uint8_t >
INLINE_GRAPHCOLORING
std::vector< T_Color > graphColoring
(
    T_Neighbors const & rvNeighbors    ,
    size_t      const & rnElements     ,
    bool        const   rbUniformColors,
    std::function< size_t( T_Neighbors const &, T_Id const & ) > const & rfGetNeighborsSize,
    /* lambdas as default arguments gives errors for gcc 4.9.1 and below and works for 4.9.3 or greater */
    /*
    #if defined ( __GNUC__ ) && ( __GNUC__ >= 5 || ( __GNUC__ == 4 && ( __GNUC_MINOR__ >= 10 || ( __GNUC_MINOR__ == 9 && __GNUC_PATCHLEVEL__ >= 3 ) ) ) )
        = []( T_Neighbors const & x, T_Id const & i ){ return x[i].size(); },
    #endif
    */
    std::function< T_Id( T_Neighbors const &, T_Id const &, size_t const & ) > const & rfGetNeighbor
    /*
    #if defined ( __GNUC__ ) && ( __GNUC__ >= 5 || ( __GNUC__ == 4 && ( __GNUC_MINOR__ >= 10 || ( __GNUC_MINOR__ == 9 && __GNUC_PATCHLEVEL__ >= 3 ) ) ) )
        = []( T_Neighbors const & x, T_Id const & i, size_t const & j ){ return x[i][j]; }
    #endif
    */
);
