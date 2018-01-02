cmake_minimum_required( VERSION 3.1 )

set( CMAKE_CXX_STANDARD 11 ) # needs CMake 3.1. Need C++11 for auto-keyword and list initializers ... I sure have gotten lazy :S

set( LEMONADE_SUBMODULE_FOLDER "./extern/LeMonADE" )
option( PULL_LEMONADE "If enabled LeMonADE will be downloaded and compiled in the folder ${LEMONADE_SUBMODULE_FOLDER}" OFF )

set( CMAKE_CXX_FLAGS "-Wall -Wextra" )
set( CMAKE_CXX_FLAGS_RELEASE "-O3 -msse2 -mssse3 -fexpensive-optimizations -DNDEBUG" )
set( CMAKE_CXX_FLAGS_DEBUG   "-g -O0" )

if( NOT CMAKE_BUILD_TYPE )
    set( CMAKE_BUILD_TYPE "Release" ) #default build type is Release
endif()
if( NOT CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT CMAKE_BUILD_TYPE STREQUAL "Release" )
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
        if ( NOT EXISTS "${PROJECT_SOURCE_DIR}/${LEMONADE_SUBMODULE_FOLDER}/lib/lib${LEMONADE_LIBRARY}.a" )
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

file( GLOB SOURCE_FILES ${PROJECT_SOURCE_DIR}/src/singleLinearChain/*.cpp )
add_executable( singleLinearChain ${SOURCE_FILES} )
target_link_libraries( singleLinearChain ${LEMONADE_LIBRARY} )

file( GLOB SOURCE_FILES ${PROJECT_SOURCE_DIR}/tests/*.cpp )
add_executable( mainTestAll ${SOURCE_FILES} )
target_link_libraries( mainTestAll ${LEMONADE_LIBRARY} )

find_package( CUDA REQUIRED )
set( CUDA_ARCH "20" CACHE STRING "E.g. 30 for Kepler or 61 for the P100. Default currently 20. To be used with -DCUDA_ARCH:STRING=30" )
message( "Compile for sm_${CUDA_ARCH}" )
# Instead of adding --compiler-bindir /usr/bin/g++-4.9 you should use cmake .. -DCUDA_HOST_COMPILER=/usr/bin/g++-4.9
set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 -lineinfo -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} -Xcompiler -fPIC )
if( CMAKE_BUILD_TYPE STREQUAL "Debug" )
    set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g )
endif()

# parallel Sc BFM
message( " CMAKE_CXX_FLAGS         = ${CMAKE_CXX_FLAGS}" )
message( " CMAKE_CXX_FLAGS_RELEASE = ${CMAKE_CXX_FLAGS_RELEASE}" )
message( " CMAKE_CXX_FLAGS_DEBUG   = ${CMAKE_CXX_FLAGS_DEBUG}" )
cuda_add_executable( SimulatorCUDAGPUScBFM_AB_Type
    src/pscBFMLegacy/UpdaterGPUScBFM_AB_Type.cu
    src/pscBFMLegacy/mainCUDASimulatorGPUScBFM_AB_Type.cpp
)
target_link_libraries( SimulatorCUDAGPUScBFM_AB_Type ${LEMONADE_LIBRARY} )

# Bcc BFM
#cuda_add_executable( GPUBccBFMMeltLinearChains
#    src/bccBFMLegacy/UpdaterGPUBccBFM.cu
#    src/bccBFMLegacy/mainCUDASimulatorMelt.cpp
#)
#target_link_libraries( GPUBccBFMMeltLinearChains ${LEMONADE_LIBRARY} )
