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
    T_Neighbors const & rvNeighbors,
    size_t      const & rnElements ,
    bool        const   rbUniformColors = false,
    std::function< size_t( T_Neighbors const &, T_Id const & ) > const & rfGetNeighborsSize =
        []( T_Neighbors const & x, T_Id const & i ){ return x[i].size(); },
    std::function< T_Id( T_Neighbors const &, T_Id const &, size_t const & ) > const & rfGetNeighbor =
        []( T_Neighbors const & x, T_Id const & i, size_t const & j ){ return x[i][j]; }
);
