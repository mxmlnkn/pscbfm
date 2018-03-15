#!/bin/bash

. ../src/pscbfm/testCorrectness.sh

sTime=$( date +%Y-%m-%dT%H-%M-%S )
folder="benchmark-$sTime"
mkdir "$folder"
(
    cd -- "$folder"
    dumpsysinfo "sysinfo-host.log"
    dumpsysinfo "sysinfo-srun.log"
)
echo "Created folder '$folder'"

# These are some groups needed in testing for findSlowdowns.txt

# First slowdown, was because of change to neighbor struct
benchmarks=(
    'benchmark-00-original'
    'benchmark-00-original-8f7cdb9'
    'benchmark-00-original-08c11d2'
    'benchmark-00-original-18fb770'
    'benchmark-01-externalCheckLattice'
)
# secon slow down, was because of change to checkFront
benchmarks=(
    'benchmark-05-hostOptimizations'
    'benchmark-05-hostOptimizations-88168bc'
    'benchmark-05-hostOptimizations-b626793'
    'benchmark-05-hostOptimizations-5e4eb2c'
    'benchmark-06-monomerPositionCalls'
)
# to find slowdown, it was in C because of the a simple change in the kernel -.-
benchmarks=(
    'benchmark-07-hostOptimizations'
    'benchmark-07-hostOptimizationsA-2f81219'
    'benchmark-07-hostOptimizationsB-428f5ca'
    'benchmark-07-hostOptimizationsC-06dc6e4'
    'benchmark-07-hostOptimizationsD-c7fd142'
    'benchmark-07-hostOptimizationsE-5301004'
    'benchmark-07-hostOptimizationsF-932a041'
    'benchmark-07-hostOptimizationsG-beb8347'
    'benchmark-07-hostOptimizationsH-e6c2c19'
    'benchmark-07-hostOptimizationsI-aa5e4e4'
    'benchmark-07-hostOptimizationsJ-5b33fc0'
    'benchmark-07-hostOptimizationsK-b92a3e6'
    'benchmark-07-hostOptimizationsL-3e422ae'
#   'benchmark-07-hostOptimizationsM-35c9303'
    'benchmark-07-hostOptimizationsN-6fd2732'
    'benchmark-07-hostOptimizationsO-7045a4d'
    'benchmark-08-saveHostNeighborCopy'
)
# For finding the problem of the contradictory cudaMemset time measurement (kernels suppsoedly slower, but tGpu faster)
benchmarks=(
    'benchmark-09-textureObjectsOnly'
    'benchmark-10-cudaMemset'
)
# finding why after between 10 and 15 nothing changes, but when removing the cudaMemset suddenly after 15 we have a 2% speedup instead of a 0.1% slowdown at 10 ...
benchmarks=(
    'benchmark-10-cudaMemset'
    'benchmark-11-gmemPolymers'
    'benchmark-12-saveChecks'
    'benchmark-13a-saveGroupIdMapping'
    'benchmark-13b-128BitLoad'
    'benchmark-14a-directionCalculation'
    'benchmark-14b-directionLocalTable'
    'benchmark-15a-checkFrontNaive'
    'benchmark-15b-checkFrontCompact'
    'benchmark-15c-checkFrondReturns'
    'benchmark-15ca-ce1407a'
    'benchmark-15d-53c2559'
    'benchmark-15d-d998266'
)

benchmarks=(
    'pscbfm'
    'benchmark-00-original'
    'benchmark-01-externalCheckLattice'
    'benchmark-02-constantsToArguments'
    'benchmark-03-textureObjectsKernel1'
    'benchmark-04-textureObjectsExceptsMonomers'
    'benchmark-05-hostOptimizations'
    'benchmark-06-monomerPositionCalls'
    'benchmark-07-hostOptimizations'
    'benchmark-08-saveHostNeighborCopy'
    'benchmark-09-textureObjectsOnly'
    'benchmark-10-cudaMemset'
    'benchmark-11-gmemPolymers'
    'benchmark-12-saveChecks'
    'benchmark-13a-saveGroupIdMapping'
    'benchmark-13b-128BitLoad'
    'benchmark-14a-directionCalculation'
    'benchmark-14b-directionLocalTable'
    'benchmark-15a-checkFrontNaive'
    'benchmark-15b-checkFrontCompact'
    'benchmark-15c-checkFrondReturns'
)
quickBenchmarks=( $( find ../benchmarks/ -mindepth 1 -maxdepth 1 -type d |
                     sort -n | sed 's|../benchmarks/|benchmark-|' ) )
for benchmark in "${quickBenchmarks[@]}"; do
    # if you want it fast, then don't use --force-compile and only do 1 benchmark run
    checkSc -b -n 1 --folder "$folder" "$benchmark"
    #checkSc -b --folder "$folder" "$benchmark"
    if [ "${benchmark#benchmark-}" != "$benchmark" ]; then
        cp --no-clobber "../benchmarks/${benchmark#benchmark-}/UpdaterGPUScBFM_AB_Type.cu" \
           "$folder/${benchmark#benchmark-}-UpdaterGPUScBFM_AB_Type.cu"
    else
        cp --no-clobber "../src/pscbfm/UpdaterGPUScBFM_AB_Type.cu" \
           "$folder/${benchmark}-UpdaterGPUScBFM_AB_Type.cu"
    fi
done

# Collect timings
benchmarks=( $( find "$folder" -maxdepth 1 -mindepth 1 -type f | sed -nr 's|^.*/(.*)-20[0-9]{2}-[01][0-9]-[0-3][0-9]T.*|\1|p' | sort -u ) )
for benchmark in "${benchmarks[@]}"; do
    echo "===== $benchmark ====="
    file=$( find "$folder" -maxdepth 1 -mindepth 1 -name "$benchmark-20[0-9][0-9]-[01][0-9]-*-kernels.log" )
    timings=( $( sed -nr 's|^(t[A-Z].*) = [0-9.]*s$|\1|p' "$file" | sort -u ) )
    for timing in "${timings[@]}"; do
        values=( $( sed -nr 's|^'"$timing"' = ([0-9.]*)s$|\1|p' "$file" ) )
        stats=( $( python3 -c '
import sys, numpy as np
x = np.array( sys.argv[1:], dtype=np.float64 )
print( np.min(x), np.mean(x), np.std(x), np.max(x) )
        ' ${values[@]} ) )
        printf '% -9s = ' "$timing"
        printf '% 8f ' ${values[@]}
        printf '=> % 8f | % 8f +- % 8f | % 8f\n' ${stats[@]}
    done
done | tee "$folder/crawledTimings.txt"
# sed -i 's|= .* => |=|' crawledTimings.txt # delete single measurements

echo '' > "$folder/allCrawledKernelTimings.txt"
for benchmark in "${benchmarks[@]}"; do
    echo "===== $benchmark =====" | tee -a "$folder/allCrawledKernelTimings.txt"
    file=$( find "$folder" -maxdepth 1 -mindepth 1 -name "$benchmark-20[0-9][0-9]-[01][0-9]-*-kernels.log" )
    sed -n '/Profiling result:/,/infile/{ s|^infile[^\n]*||; p; }' "$file" >> "$folder/allCrawledKernelTimings.txt"
    # only print the first result for the comparison file, the measurements are quite exact, as you can convince yoursef in "$benchmark-kernelTimings.txt"
    sed -n '/Profiling result:/,/infile/{ /infile/{ s|^infile[^\n]*||; q; }; p; }' "$file"
done | tee "$folder/crawledKernelTimings.txt"
