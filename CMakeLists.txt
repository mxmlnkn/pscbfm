cmake_minimum_required( VERSION 3.1 )

set( CMAKE_CXX_STANDARD 11 ) # needs CMake 3.1. Need C++11 for auto-keyword and list initializers ... I sure have gotten lazy :S

find_path( LEMONADE_INCLUDE_DIR
    NAMES
        LeMonADE/core/ConfigureSystem.h
        LeMonADE/core/Ingredients.h
        LeMonADE/utility/TaskManager.h
    PATHS
        ENV LEMONADE_ROOT
        ENV LEMONADE_INCLUDE_DIR
        /usr/local/include
        /usr/include
    PATH_SUFFIXES
        lemonade
        LeMonADE
        lemonade/include
        LeMonADE/include
        include
    DOC "LeMonADE include location"
)
set( LEMONADE_LIBRARY "LeMonADE" )
find_path( LEMONADE_LIBRARY_DIR
    NAMES
        "lib${LEMONADE_LIBRARY}.a"
    PATHS
        ENV LEMONADE_ROOT
        ENV LEMONADE_LIBRARY_DIR
        /usr/local
        /usr
    PATH_SUFFIXES
        build/lib
        lib
    DOC "LeMonADE library location"
)

if( NOT DEFINED LEMONADE_INCLUDE_DIR )
    message( "LEMONADE_INCLUDE_DIR is not provided. If build fails, use -DLEMONADE_INCLUDE_DIR=/path/to/LeMonADE/headers/ or install to default location" )
endif()

if( NOT DEFINED LEMONADE_LIBRARY_DIR )
    message( "LEMONADE_LIBRARY_DIR is not provided. If build fails, use -DLEMONADE_LIBRARY_DIR=/path/to/LeMonADE/lib/ or install to default location" )
endif()

# Declare include SYSTEM, in order to suppress C++11-induced warnings from
# LeMonADE and Loki headers: warning:
# ‘template<class> class std::auto_ptr’ is deprecated [-Wdeprecated-declarations]
include_directories( SYSTEM ${LEMONADE_INCLUDE_DIR} )
link_directories   ( SYSTEM ${LEMONADE_LIBRARY_DIR} )

file( GLOB SOURCE_FILES ${PROJECT_SOURCE_DIR}/src/singleLinearChain/*.cpp )
add_executable( singleLinearChain ${SOURCE_FILES} )
target_link_libraries( singleLinearChain ${LEMONADE_LIBRARY} )

file( GLOB SOURCE_FILES ${PROJECT_SOURCE_DIR}/tests/*.cpp )
add_executable( mainTestAll ${SOURCE_FILES} )
target_link_libraries( mainTestAll ${LEMONADE_LIBRARY} )

