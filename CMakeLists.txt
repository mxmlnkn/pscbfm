cmake_minimum_required( VERSION 3.1 )

set( CMAKE_CXX_STANDARD 11 ) # needs CMake 3.1. Need C++11 for auto-keyword and list initializers ... I sure have gotten lazy :S


set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -msse2 -mssse3 -fexpensive-optimizations" )
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wall -Wextra -DNDEBUG" )

#define value of CMAKE_BUILD_TYPE depending on input
if( NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Release" )
    set( CMAKE_BUILD_TYPE "Release" ) #default build type is Release
elseif( CMAKE_BUILD_TYPE STREQUAL "Debug" )
    set( CMAKE_BUILD_TYPE "Debug" )
else()
    message( FATAL_ERROR "Invalid build type ${CMAKE_BUILD_TYPE} specified." )
endif()

#output depending on build type
if( CMAKE_BUILD_TYPE STREQUAL "Release" )
    set( CMAKE_VERBOSE_MAKEFILE 0 )
    message( "Build type is ${CMAKE_BUILD_TYPE}" )
    message( "USING CXX COMPILER FLAGS ${CMAKE_CXX_FLAGS_RELEASE}" )
elseif( CMAKE_BUILD_TYPE STREQUAL "Debug" )
    set( CMAKE_VERBOSE_MAKEFILE 1 )
    message( "Build type is ${CMAKE_BUILD_TYPE}" )
    message( "USING CXX COMPILER FLAGS ${CMAKE_CXX_FLAGS_DEBUG}" )
endif()



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

