cmake_minimum_required( VERSION 3.1 )

set( CMAKE_CXX_STANDARD 11 ) # needs CMake 3.1. Need C++11 for auto-keyword and list initializers ... I sure have gotten lazy :S

set( LEMONADE_SUBMODULE_FOLDER "./extern/LeMonADE" )
option( PULL_LEMONADE "If enabled LeMonADE will be downloaded and compiled in the folder ${LEMONADE_SUBMODULE_FOLDER}" OFF )

set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -msse2 -mssse3 -fexpensive-optimizations" )
set( CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -O0 -Wall -Wextra -DNDEBUG" )

if( NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Release" )
    set( CMAKE_BUILD_TYPE "Release" ) #default build type is Release
elseif( CMAKE_BUILD_TYPE STREQUAL "Debug" )
    set( CMAKE_BUILD_TYPE "Debug" )
else()
    message( FATAL_ERROR "Invalid build type ${CMAKE_BUILD_TYPE} specified." )
endif()

if( CMAKE_BUILD_TYPE STREQUAL "Release" )
    set( CMAKE_VERBOSE_MAKEFILE 0 )
    message( "Build type is ${CMAKE_BUILD_TYPE}" )
    message( "USING CXX COMPILER FLAGS ${CMAKE_CXX_FLAGS_RELEASE}" )
elseif( CMAKE_BUILD_TYPE STREQUAL "Debug" )
    set( CMAKE_VERBOSE_MAKEFILE 1 )
    message( "Build type is ${CMAKE_BUILD_TYPE}" )
    message( "USING CXX COMPILER FLAGS ${CMAKE_CXX_FLAGS_DEBUG}" )
endif()


# find LeMonADE headers and compiled library
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
        ${PROJECT_SOURCE_DIR}/extern
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
        ${PROJECT_SOURCE_DIR}/extern
    PATH_SUFFIXES
        LeMonADE/lib
        build/lib
        lib
    DOC "LeMonADE library location"
)

if( NOT LEMONADE_INCLUDE_DIR )
    message( "LEMONADE_INCLUDE_DIR is not provided. If build fails, use -DLEMONADE_INCLUDE_DIR=/path/to/LeMonADE/headers/ or install to default location" )
endif()
if( NOT LEMONADE_LIBRARY_DIR )
    message( "LEMONADE_LIBRARY_DIR is not provided. If build fails, use -DLEMONADE_LIBRARY_DIR=/path/to/LeMonADE/lib/ or install to default location" )
endif()
if( ( NOT LEMONADE_INCLUDE_DIR ) OR ( NOT LEMONADE_LIBRARY_DIR ) )
    if( PULL_LEMONADE )
        message( "Installing LeMonADE ..." )
        # every command runs in its own process, that's why this workaround
        # of calling sh with a script was used
        execute_process(
            # installing into the source is very unclean, but as 80% is
            # header only it will simply result in only the compiled
            # library being 'installed'
            COMMAND sh -c "\
            git submodule update --init -- ${LEMONADE_SUBMODULE_FOLDER}     && \
            cd ${LEMONADE_SUBMODULE_FOLDER}                                 && \
            mkdir -p build install                                          && \
            cd build                                                        && \
            cmake -DINSTALLDIR_LEMONADE=.. DCMAKE_INSTALL_PREFIX=.. ..      && \
            make install                                                       \
            "
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        )
        if ( NOT EXISTS "${PROJECT_SOURCE_DIR}/${LEMONADE_SUBMODULE_FOLDER}/lib/lib${LEMONADE_LIBRARY}" )
            message( FATAL_ERROR "Something went wrong when trying to install LeMonADE from source!" )
        endif()
        set( LEMONADE_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/${LEMONADE_SUBMODULE_FOLDER}/include" )
        set( LEMONADE_LIBRARY_DIR "${PROJECT_SOURCE_DIR}/${LEMONADE_SUBMODULE_FOLDER}/lib" )
    else()
        message( "You can also call cmake again with the option -DPULL_LEMONADE=ON in order to try to automatically download and compile LeMonADE into a default subfolder." )
    endif()
else()
    message( "Found LeMonADE at ${LEMONADE_INCLUDE_DIR} and ${LEMONADE_LIBRARY_DIR}" )
endif()



# Declare include SYSTEM, in order to suppress C++11-induced warnings from
# LeMonADE and Loki headers: warning:
# ‘template<class> class std::auto_ptr’ is deprecated [-Wdeprecated-declarations]
include_directories( SYSTEM ${LEMONADE_INCLUDE_DIR} )
link_directories   ( SYSTEM ${LEMONADE_LIBRARY_DIR} )

#file( GLOB SOURCE_FILES ${PROJECT_SOURCE_DIR}/src/singleLinearChain/*.cpp )
#add_executable( singleLinearChain ${SOURCE_FILES} )
#target_link_libraries( singleLinearChain ${LEMONADE_LIBRARY} )
#
#file( GLOB SOURCE_FILES ${PROJECT_SOURCE_DIR}/tests/*.cpp )
#add_executable( mainTestAll ${SOURCE_FILES} )
#target_link_libraries( mainTestAll ${LEMONADE_LIBRARY} )

add_subdirectory(projects)
