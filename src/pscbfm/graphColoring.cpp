#include "graphColoring.hpp"
#include "graphColoring.tpp"

#include <vector>


#define INSTANTIATE_TMP( T_Neighbors, T_Id, T_Color )                                                \
template std::vector< T_Color > graphColoring< T_Neighbors >                                         \
(                                                                                                    \
    T_Neighbors const & rvNeighbors,                                                                 \
    size_t      const & rnElements ,                                                                 \
    std::function< size_t( T_Neighbors const &, T_Id const & ) > const & rfGetNeighborsSize,         \
    std::function< T_Id( T_Neighbors const &, T_Id const &, size_t const & ) > const & rfGetNeighbor \
);
/* vector / list for each monomer */
INSTANTIATE_TMP( std::vector< std::vector< uint8_t > >, uint8_t , uint8_t  )
INSTANTIATE_TMP( std::vector< std::vector< uint8_t > >, uint8_t , uint16_t )
INSTANTIATE_TMP( std::vector< std::vector< uint8_t > >, uint8_t , uint32_t )
INSTANTIATE_TMP( std::vector< std::vector< uint8_t > >, uint16_t, uint8_t  )
INSTANTIATE_TMP( std::vector< std::vector< uint8_t > >, uint16_t, uint16_t )
INSTANTIATE_TMP( std::vector< std::vector< uint8_t > >, uint16_t, uint32_t )
INSTANTIATE_TMP( std::vector< std::vector< uint8_t > >, uint32_t, uint8_t  )
INSTANTIATE_TMP( std::vector< std::vector< uint8_t > >, uint32_t, uint16_t )
INSTANTIATE_TMP( std::vector< std::vector< uint8_t > >, uint32_t, uint32_t )

#undef INSTANTIATE_TMP
