cmake_minimum_required( VERSION 3.1 )

set( CMAKE_CXX_STANDARD 11 ) # needs CMake 3.1. Need C++11 for auto-keyword and list initializers ... I sure have gotten lazy :S

set( LEMONADE_SUBMODULE_FOLDER "./extern/LeMonADE" )
option( PULL_LEMONADE "If enabled LeMonADE will be downloaded and compiled in the folder ${LEMONADE_SUBMODULE_FOLDER}" OFF )
option( BUILD_BENCHMARKS "If enabled possible tens of versions will be compiled in order to compare them" OFF )

set( CMAKE_CXX_FLAGS "-Wall -Wextra" )
set( CMAKE_CXX_FLAGS_RELEASE "-O3 -msse2 -mssse3 -fexpensive-optimizations -DNDEBUG" )
set( CMAKE_CXX_FLAGS_DEBUG   "-g -O0" )
message( "CUDA_HOST_COMPILER = ${CUDA_HOST_COMPILER}")
if( CUDA_HOST_COMPILER )
    message( "Custom CUDA-Host compiler chosen. In order to ensure ABI compatibility we have to use the same compiler for non-CUDA code." )
    message( "If you haven't done so, then you also need to compile LeMonADE with the chosen compiler!" )
    message( "E.g. use: cmake -D CMAKE_CXX_COMPILER <cpp compiler path> .." )
    # Warn on a time consuming known problem, i.e. linker errors when trying to link C++ code with gcc instead of g++
    # this happens, because it seems that module environment was configured to set CUDA_HOST_COMPILER environment variable and it seems that CMake actually searches for options in environment variables ...then again this doesn't seem to be the case, neither is cmake aliased Oo? where is this coming from ...
    if ( "${CUDA_HOST_COMPILER}" MATCHES ".*/gcc[^/]*$" )
        message( WARNING "CUDA_HOST_COMPILER must not be a gcc compiler, use g++ instead!" )
        string( REGEX REPLACE "(.*)/gcc([^/]*)$" "\\1/g++\\2" CUDA_HOST_COMPILER ${CUDA_HOST_COMPILER} )
        message( STATUS "Changed CUDA_HOST_COMPILER to '${CUDA_HOST_COMPILER}'" )
    endif()
    set( CMAKE_CXX_COMPILER ${CUDA_HOST_COMPILER} )
    # set( CMAKE_CXX_COMPILER ${CUDA_HOST_COMPILER} ) # could try to substitute g++ with gcc but would be unsecure, plus we don't really need it I think.
endif()
if ( "${CMAKE_CXX_COMPILER}" MATCHES ".*/gcc[^/]*$" )
    message( "CMAKE_CXX_COMPILER must not be a gcc compiler, use g++ instead!" )
    string( REGEX REPLACE "(.*)/gcc([^/]*)$" "\\1/g++\\2" CMAKE_CXX_COMPILER ${CMAKE_CXX_COMPILER} )
    message( STATUS "Changed CMAKE_CXX_COMPILER to '${CMAKE_CXX_COMPILER}'" )
endif()

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

if ( NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/Fundamental/Fundamental.hpp" )
    execute_process( COMMAND git submodule update --recursive )
endif()

# find LeMonADE headers and compiled library
find_path( LEMONADE_INCLUDE_DIR
    NAMES
        LeMonADE/core/ConfigureSystem.h
        LeMonADE/core/Ingredients.h
        LeMonADE/utility/TaskManager.h
    PATHS
        ${PROJECT_SOURCE_DIR}/extern
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
        ${PROJECT_SOURCE_DIR}/extern
        ENV LEMONADE_ROOT
        ENV LEMONADE_LIBRARY_DIR
        /usr/local
        /usr
    PATH_SUFFIXES
        LeMonADE/lib
        lib
        build/lib
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
            cmake -DCMAKE_CXX_COMPILER='${CMAKE_CXX_COMPILER}'                 \
                  -DINSTALLDIR_LEMONADE=.. DCMAKE_INSTALL_PREFIX=.. ..      && \
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

include_directories( "extern/" ) # Fundamental
add_executable( findMaxNeighbors ../src/findMaxNeighbors.cpp )
target_link_libraries( findMaxNeighbors )


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
message( "CUDA_INCLUDE_DIR " )
message( "Compile for sm_${CUDA_ARCH}" )
# Instead of adding --compiler-bindir /usr/bin/g++-4.9 you should use cmake .. -DCUDA_HOST_COMPILER=/usr/bin/g++-4.9
set( MINIMAL_CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -std=c++11 -Xcompiler -fPIC
    -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}
)
set( BENCHMARK_CUDA_NVCC_FLAGS
    # Show resource usage such as registers and memory of the GPU code. => will be in make log in benchmarking logs, although I guess this might be a reason to force recompilation before benchmarking
    -res-usage
    # Generate line-number information for device code. (for nvvp)
    --generate-line-info
    # Interleave source in PTX. May only be used in conjunction with --device-debug or --generate-line-info.
    --source-in-ptx
    # https://stackoverflow.com/questions/12388207/interpreting-output-of-ptxas-options-v
    --ptxas-options=-v
    # keep intermediary source files (ptx code). Each compilation should set their own --keep-dir!
    --keep
)
if( NOT ${CUDA_VERSION_MAJOR} LESS 9 )
    # option does not exist in CUDA 7.0.27
    # http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#ptxas-options (CUDA 9.1.85 docs)
    set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xptxas --warn-on-double-precision-use,--warn-on-local-memory-usage,--warn-on-spills )
endif()
if( CMAKE_BUILD_TYPE STREQUAL "Debug" )
    set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g )
endif()

# no find_package for thrust needed, as it should be in the include folder for CUDA

message( " CMAKE_CXX_FLAGS         = ${CMAKE_CXX_FLAGS}" )
message( " CMAKE_CXX_FLAGS_RELEASE = ${CMAKE_CXX_FLAGS_RELEASE}" )
message( " CMAKE_CXX_FLAGS_DEBUG   = ${CMAKE_CXX_FLAGS_DEBUG}" )


# parallel Sc BFM
set( BIN_NAME "pscbfm" )
set( GPU_SOURCE_FOLDER "${CMAKE_BINARY_DIR}/gpu-sources/${BIN_NAME}" )
file( MAKE_DIRECTORY "${GPU_SOURCE_FOLDER}" )
set( CUDA_NVCC_FLAGS ${MINIMAL_CUDA_NVCC_FLAGS} ${BENCHMARK_CUDA_NVCC_FLAGS} "--keep-dir=${GPU_SOURCE_FOLDER}" )
cuda_add_executable( ${BIN_NAME}
    src/pscbfm/UpdaterGPUScBFM_AB_Type.cu
    src/pscbfm/mainCUDASimulatorGPUScBFM_AB_Type.cpp
)
target_link_libraries( ${BIN_NAME} ${LEMONADE_LIBRARY} )


# Bcc BFM
#cuda_add_executable( GPUBccBFMMeltLinearChains
#    src/bccBFMLegacy/UpdaterGPUBccBFM.cu
#    src/bccBFMLegacy/mainCUDASimulatorMelt.cpp
#)
#target_link_libraries( GPUBccBFMMeltLinearChains ${LEMONADE_LIBRARY} )


# CLI for coloring a BFM file
add_executable( colorBFM src/pscbfm/mainColorBFM.cpp )
target_link_libraries( colorBFM ${LEMONADE_LIBRARY} )


# minimalistic gpuinfo tool printing all properties
set_source_files_properties( src/gpuinfo.cpp PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ )
set( CUDA_NVCC_FLAGS ${MINIMAL_CUDA_NVCC_FLAGS} )
cuda_add_executable( gpuinfo src/gpuinfo.cpp )
target_link_libraries( gpuinfo )



### Benchmarks:

# https://stackoverflow.com/questions/7787823/cmake-how-to-get-the-name-of-all-subdirectories-of-a-directory

macro( subdirlist result curdir )
    file( GLOB children RELATIVE ${curdir} ${curdir}/* )
    set( dirlist "" )
    foreach( child ${children} )
        if( IS_DIRECTORY ${curdir}/${child} )
            list( APPEND dirlist ${child} )
        endif()
    endforeach()
    set( ${result} ${dirlist} )
endmacro()

if( BUILD_BENCHMARKS )
    # don't need warnings for the older versions, don't gonna optimize them again in respect to warnings
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w" )
    set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -w" )
    message( "=== Configuring Benchmarks ===" )
    subdirlist( lsubdirs "${PROJECT_SOURCE_DIR}/benchmarks" ) # must be full path!
    foreach( subdir ${lsubdirs} )
        message( "Configuring benchmarks/${subdir}" )
        set( BIN_NAME "${subdir}" )
        set( GPU_SOURCE_FOLDER "${CMAKE_BINARY_DIR}/gpu-sources/${BIN_NAME}" )
        file( MAKE_DIRECTORY "${GPU_SOURCE_FOLDER}" )
        set( CUDA_NVCC_FLAGS ${MINIMAL_CUDA_NVCC_FLAGS} ${BENCHMARK_CUDA_NVCC_FLAGS} "--keep-dir=${GPU_SOURCE_FOLDER}" )
        cuda_add_executable( benchmark-${subdir}
            "benchmarks/${subdir}/UpdaterGPUScBFM_AB_Type.cu"
            "benchmarks/${subdir}/mainCUDASimulatorGPUScBFM_AB_Type.cpp"
        )
        target_link_libraries( benchmark-${subdir} ${LEMONADE_LIBRARY} )
    endforeach()
endif()
