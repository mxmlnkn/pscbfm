#include "graphColoring.hpp"
#include "graphColoring.tpp"

#include <vector>


#define INSTANTIATE_TMP( T_Neighbors )                                                               \
template std::vector< T_Color > graphColoring< T_Neighbors >                                         \
(                                                                                                    \
    T_Neighbors const & rvNeighbors,                                                                 \
    size_t      const & rnElements ,                                                                 \
    std::function< size_t( T_Neighbors const &, T_Id const & ) > const & rfGetNeighborsSize,         \
    std::function< T_Id( T_Neighbors const &, T_Id const &, size_t const & ) > const & rfGetNeighbor \
);

INSTANTIATE_TMP( std::vector< std::vector< uint8_t > > )

#undef INSTANTIATE_TMP
