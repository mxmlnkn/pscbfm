= Known Bugs =

When g++ 7 is installed and used:
    /usr/bin/ld: CMakeFiles/SimulatorCUDAGPUScBFM_AB_Type.dir/src/pscBFMLegacy/SimulatorCUDAGPUScBFM_AB_Type_generated_UpdaterGPUScBFM_AB_Type.cu.o: relocation R_X86_64_32S against `.bss' can not be used when making a shared object; recompile with -fPIC
    /usr/bin/ld: final link failed: Nonrepresentable section on output
Workaround:
    Add -fPIC to CUDA_NVCC_FLAGS in CMakeLists.txt ... not sure why suddenly though. Worked all this time


/sw/taurus/libraries/cuda/7.0.28/bin/nvcc /home/user/Master/mt-progwork/src/pscBFMLegacy/UpdaterGPUScBFM_AB_Type.cu -c -o /home/user/Master/mt-progwork/build/CMakeFiles/SimulatorCUDAGPUScBFM_AB_Type.dir/src/pscBFMLegacy/./SimulatorCUDAGPUScBFM_AB_Type_generated_UpdaterGPUScBFM_AB_Type.cu.o -ccbin /sw/global/compilers/gcc/4.9.1/bin/gcc -m64 -Xcompiler ,\"-Wall\",\"-Wextra\",\"-O3\",\"-msse2\",\"-mssse3\",\"-fexpensive-optimizations\",\"-DNDEBUG\" -std=c++11 -lineinfo -gencode arch=compute_35,code=sm_35 -Xcompiler -fPIC -DNVCC -I/sw/taurus/libraries/cuda/7.0.28/include -I/home/user/Master/mt-progwork/extern/LeMonADE/include -I/sw/taurus/libraries/cuda/7.0.28/include
/home/user/Master/mt-progwork/src/pscBFMLegacy/graphColoring.h:19:17: error: default argument for template parameter for class enclosing ‘<lambda>’
         []( T_Neighbors const & x, T_Id const & i ){ return x[i].size(); },
                 ^
=> works with 4.9.3, but not with 4.9.1 ...
