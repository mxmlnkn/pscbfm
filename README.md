# Compilation

    git clone --recursive git@bitbucket.org:mxmlnkn/mt-progwork.git
    cd mt-progwork && mkdir build && cd build
    cmake .. -DCUDA_ARCH:STRING=35 && make -j 2

Tested with:

 - `GIT 1.7.1`, `GNU make 3.8.1`, `CMake 3.9.0`, `g++ 5.3.0`, `CUDA 9.0.176`
 - `GIT 1.7.1`, `GNU make 3.8.1`, `CMake 3.3.1`, `g++ 4.8.0`, `CUDA 7.0.28`
 - Does not work with gcc older than `4.8.0`, because those versions can't cope with some initializer lists. See also CMake [`cxx_generalized_initializers`](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2672.htm).
 - Does not work with gcc older than `4.7.0`, because `-std=c++11` was added in `4.7.0`

Note that Fermie `compute_20` is deprecated since CUDA 9.

In order to understand what is behind the CMake procedure you can also try to compile the main program manually with:

    cd build
    CCOPTS='-Xcompiler -Wall,-Wextra,-O3,-msse2,-mssse3,-fexpensive-optimizations -DNDEBUG -std=c++11'
    INCLUDES='-I../extern -isystem ../extern/LeMonADE/include '
    nvcc $INCLUDES $CCOPTS -c -o mainCUDASimulatorGPUScBFM_AB_Type.cpp.o ../src/pscbfm/mainCUDASimulatorGPUScBFM_AB_Type.cpp
    nvcc $INCLUDES $CCOPTS -c -o UpdaterGPUScBFM_AB_Type.cu.o -Xcompiler -fPIC -gencode arch=compute_30,code=sm_30 ../src/pscbfm/UpdaterGPUScBFM_AB_Type.cu
    nvcc $CCOPTS -o pscbfm mainCUDASimulatorGPUScBFM_AB_Type.cpp.o UpdaterGPUScBFM_AB_Type.cu.o -L../extern/LeMonADE/lib --cudart static -lpthread -ldl -lrt -lLeMonADE -lcuda -Xcompiler -static-libstdc++,-static-libgcc

You can try a test run with e.g.:

    ./pscbfm -i ../tests/inputPscBFM.bfm -e ../tests/resultNormPscBFM.seeds -o result.bfm -m 10000 -s 10000

A more extensive test routine is available with `checkSc` in ``. E.g. to benchmark and profile everything on Taurus do:

    . ../src/pscbfm/testCorrectness.sh
    checkSc -b -p --force-compile --folder "$folder" --batchjob-prefix 'srun --cpu-freq=highp1 --gpufreq=2505:692' --arch 35 "$benchmark"

See also `benchmarkK80.pbs`. Most options are optional: on a system without the need for SLURM or PBS, you might already have success with `checkSc`, which only runs the simulation once checking for the correctness, `checkSc -b`, which runs the simulation multiple times to get a statistic for the manual counters, or with `check -p`, which does all the above and also creates extensive and time expensive profiling data files.


# Known Bugs

## When g++ 7 is installed and used:

    /usr/bin/ld: CMakeFiles/SimulatorCUDAGPUScBFM_AB_Type.dir/src/pscBFMLegacy/SimulatorCUDAGPUScBFM_AB_Type_generated_UpdaterGPUScBFM_AB_Type.cu.o: relocation R_X86_64_32S against '.bss' can not be used when making a shared object; recompile with -fPIC
    /usr/bin/ld: final link failed: Nonrepresentable section on output

### Workaround:

 - Add `-fPIC` to `CUDA_NVCC_FLAGS` in `CMakeLists.txt`. Can't say since which version this error started appearing


## Compilation of graphColoring.h fails:

    /sw/taurus/libr aries/cuda/7.0.28/bin/nvcc ~/mt-progwork/src/pscBFMLegacy/UpdaterGPUScBFM_AB_Type.cu -c -o ~/mt-progwork/build/CMakeFiles/SimulatorCUDAGPUScBFM_AB_Type.dir/src/pscBFMLegacy/./SimulatorCUDAGPUScBFM_AB_Type_generated_UpdaterGPUScBFM_AB_Type.cu.o -ccbin /sw/global/compilers/gcc/4.9.1/bin/gcc -m64 -Xcompiler ,\"-Wall\",\"-Wextra\",\"-O3\",\"-msse2\",\"-mssse3\",\"-fexpensive-optimizations\",\"-DNDEBUG\" -std=c++11 -lineinfo -gencode arch=compute_35,code=sm_35 -Xcompiler -fPIC -DNVCC -I/sw/taurus/libraries/cuda/7.0.28/include -I~/mt-progwork/extern/LeMonADE/include -I/sw/taurus/libraries/cuda/7.0.28/include
    ~/mt-progwork/src/pscBFMLegacy/graphColoring.h:19:17: error: default argument for template parameter for class enclosing ‘<lambda>’
             []( T_Neighbors const & x, T_Id const & i ){ return x[i].size(); },
                     ^

### Workaround:

 - works with 4.9.3, but not with 4.9.1
 - for sake of easier use this works now for older compilers by not providing default functors


## Changing the major compiler version leads to problems:

 - LeMonADE needs to be built with the same version of `g++`, e.g. when compiling LeMonADE with g++ 5, but pscbfm with g++ 4.9, segfaults which are very time-consuming to track down will appear because of ABI incompatibilities.
 - CMake doesn't yet automatically check for this and rebuild LeMonADE if it finds it uses a different version

### Workaround:

 - Remake LeMonADE with the current compiler version: `cd extern/LeMonADE/; rm -r build; mkdir build; cd build; cmake -DINSTALLDIR_LEMONADE=.. DCMAKE_INSTALL_PREFIX=.. ..; make install`
 - You can also just delete the built library and call cmake again to let it rebuild LeMonADE automatically as long as the `PULL_LEMONADE` option is set to `ON`: `rm ../extern/LeMonADE/lib/libLeMonADE.a; cmake ..`

## Call of overloaded swap is ambiguous

    In file included from /sw/global/compilers/gcc/5.3.0/include/c++/5.3.0/deque:64:0,
                     from /sw/global/compilers/gcc/5.3.0/include/c++/5.3.0/stack:60,
                     from /home/user/progwork/src/AnalyzerMsd.tpp:7,
                     from /home/user/progwork/src/singleLinearChain/main.cpp:38:
    /sw/global/compilers/gcc/5.3.0/include/c++/5.3.0/bits/stl_deque.h:579:8: error: call of overloaded ‘swap(std::_Deque_base<GraphIteratorDepthFirst<Molecules<Loki::GenLinearHierarchy<Loki::Typelist<MonomerAttributeTag, Loki::NullType>, FeatureHolder, Vector3D<int> >, 7u, int>, alwaysTrue>::IteratorPosition, std::allocator<GraphIteratorDepthFirst<Molecules<Loki::GenLinearHierarchy<Loki::Typelist<MonomerAttributeTag, Loki::NullType>, FeatureHolder, Vector3D<int> >, 7u, int>, alwaysTrue>::IteratorPosition> >::iterator&, std::_Deque_base<GraphIteratorDepthFirst<Molecules<Loki::GenLinearHierarchy<Loki::Typelist<MonomerAttributeTag, Loki::NullType>, FeatureHolder, Vector3D<int> >, 7u, int>, alwaysTrue>::IteratorPosition, std::allocator<GraphIteratorDepthFirst<Molecules<Loki::GenLinearHierarchy<Loki::Typelist<MonomerAttributeTag, Loki::NullType>, FeatureHolder, Vector3D<int> >, 7u, int>, alwaysTrue>::IteratorPosition> >::iterator&)’ is ambiguous
        swap(this->_M_finish, __x._M_finish);

### Workaround

    Try to use `-D__STL_CONFIG_H`
