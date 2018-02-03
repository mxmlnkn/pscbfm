#!/bin/bash

commandExists() {
    # http://stackoverflow.com/questions/592620/check-if-a-program-exists-from-a-bash-script
    command -v "$1" > /dev/null 2>&1;
}

dumpsysinfo()
{
    local file='sysinfo.log'
    if [ -n "$1" ]; then file=$1; fi
    touch "$file"
    local command commands=(
        'ifconfig'
        'ip route'
        'ip addr show'
        'uname -a'              # includes hostname as second word on line
        'lscpu'
        'lsblk'
        'lsusb'
        'lspci'
        'lspci -v'
        'lspci -t'
        'mount'
        'ps aux'
        'cat /proc/meminfo'
        'cat /proc/cpuinfo'
        'cat /proc/uptime'
        'cat /etc/hosts'
        'nvidia-smi'
        'nvidia-smi -q'
        'git log --oneline'
        'cmake --version'
        'g++ --version'
        'nvcc --version'
    )
    local path paths=(
        ''
        '/usr/local/bin/'
        '/usr/local/sbin/'
        '/usr/bin/'
        '/usr/sbin/'
        'bin/'
        'sbin/'
    )
    for command in "${commands[@]}"; do
        for path in "${paths[@]}"; do
            if commandExists $path$command; then
                echo -e "\n===== $path$command =====\n" >> "$file"
                $path$command 2>&1 >> "$file"
                break
            fi
        done
    done
}

ignoreBFMLines()
{
    sed -r '/^#!version=[0-9.]+$/d;
            /^#[A-Z][a-z]{2} +[A-Z][a-z]{2} +[0-9]+ [0-9:]+ +[0-9]+$/d;
            /^# FeatureExcluded/d;
            /^#[ \t]+max connectivity: /d;'
    # For example one diff was:
    #   2c2
    #   < #!version=2.0
    #   ---
    #   > #!version=1.1
    #   4c4
    #   < #Sat Nov 25 13:59:41 2017
    #   ---
    #   > #Thu Oct 19 21:41:27 2017
    #   20c20
    #   < # FeatureExcludedVolumeSc<FeatureLattice<bool> >
    #   ---
    #   > # FeatureExcludedVolume<FeatureLattice<bool> >
    # version does not matter, date also not and the Feature change also not,
    # because this feature is the same but was only renamed!
    # we might even strip all comments starting with # in the future
    #
    # Not sure when exactly this appeared, but MaxConnectivity also doesn't really matter to me
    #   18c18
    #   < 00000130  69 74 79 3a 20 34 0a 23  20 46 65 61 74 75 72 65  |ity: 4.# Feature|
    #   ---
    #   > 00000130  69 74 79 3a 20 38 0a 23  20 46 65 61 74 75 72 65  |ity: 8.# Feature|
}

if ! commandExists colordiff; then alias colordiff=diff; fi

# Note: Differences to testCorrectness.sh: remove -g 1, change CUDA_ARCH to 35, remove -DCUDA_HOST_COMPILER add sruns before executions
checkSc()
( # must be () not {} or else set +x won't get reset to prior value
  # plus () means we could save the 'local' keyword
    local folder="benchmark-$( date +%Y-%m-%dT%H-%M-%S )"
    local benchmark=0
    local csrun=
    local check=
    local ctype='Release'
    local name='pscbfm'
    local iGpu=
    local arch=30
    local forceCompile=

    while [ $# -gt 0 ]; do case "$1" in
        --arch) shift; arch=$1; ;;
        -f|--force-compile) forceCompile='-B'; ;;
        -b|--benchmark) benchmark=1; ;;
        -c|--check) check='cuda-memcheck --leak-check full'; ;;
        -g|--gpu) shift; iGpu=$1; ;;
        --folder) shift; folder=$1; ;;
        --batchjob-prefix)
            shift
            if commandExists $1; then
                csrun=$1
            else
                echo "\e[36m'$1' is not a valid command, ignoring argument to--batchjob-command.\e[0m" 1>&2
            fi
            ;;
        'Debug') ctype='Debug'; ;;
        'Release') ctype='Release'; ;;
        *) name=$1; ;;
    esac; shift; done

    local sTime=$( date +%Y-%m-%dT%H-%M-%S )
    local logName="./$folder/$name-$sTime"
    mkdir -p "$folder"
    'cp' "./$name" "$folder" # archive binary for possibly dissassembly

    set -x # log commands, so that I we can easily rerun some of them
    {
        # as long as we don't delete the CMake cache the user can configure CUDA_HOST_COMPILER by itself and it will be kept even after subsequent cmake calls like the fllowing
        # -DCUDA_HOST_COMPILER=/usr/bin/g++-4.9
        cmake -DCMAKE_BUILD_TYPE="$ctype" \
              -DCUDA_ARCH:STRING="$arch"  \
              -DBUILD_BENCHMARKS=$( if [ "${name#benchmark-}" != "$name" ]; then echo ON; else echo OFF; fi ) \
              -DPULL_LEMONADE=ON .. || return 1
        # forcing make with -B might be useful in order to save output of nvcc --res-usage into logs
        # without VERBOSE=1 the output of -res-usage becomes useless as it isn't known from which compilation it came from
        # but for the benchmark script a -B would result remaking all exes and then each time running only one of them ...
        # Might not use csrun, because I think this might give problems like the binary being created and not being synced fast enough, then again, the command is executed on the "other" node anyway, so not using csrun should lead to that problem, not not using it ...
        'rm' -f "./$name" # safety so wie don't benchmark a older version accidentally
        $csrun make $forceCompile VERBOSE=1 --output-sync -j $( nproc --all ) "$name" || return 1
        # about return 1: note that because of the piping to tee this does not affect the checkSc function as all this is called in a subshell -> need to check PIPESTATUS
        mkdir -p "$folder/gpu-source/"
        'mv' "./gpu-sources/${name#benchmark-}" "$folder/gpu-source/"
    } 2>&1 | tee "$logName-build.log"
    if [ ! "${PIPESTATUS[0]}" -eq 0 ]; then return 1; fi

    # try to find GPU not running an X-Server and take that as default
    # This needs gpuinfo binary, therefore can only come after 'make'
    if [ -z "$iGpu" ]; then
        if [ ! -x './gpuinfo' ]; then
            iGpu=0
            echo "\e[36m'gpuinfo' not found or not executable, plase check '$logName-build.log'. Can't check whether GPU $iGpu runs an X-Server without it\e[0m" 1>&2
        else
            while read line; do
                if [ "${line#* }" == 'false' ]; then iGpu=${line% *}; break; fi
            done < <( ./gpuinfo | sed -nr '
                s|.*Device Number ([0-9]+).*|\1|p;
                s|.*Kernel Timeout Enabled[ \t]*: ([a-z]+)$|\1|p;' |
                sed -r -z 's|([0-9])\n|\1 |g' )
            if [ -z "$iGpu" ]; then iGpu=0; fi
        fi
    fi

    'rm' result.bfm
    # https://savannah.gnu.org/support/?109403
    #time \#
    $csrun $check "./$name" -i ../tests/inputPscBFM.bfm -e ../tests/resultNormPscBFM.seeds \
        -o result.bfm -g $iGpu -m 1000 -s 1000 2>&1 | tee "$logName-check.log"
    # cp result{,-norm}.bfm

    diff -q <( cat '../tests/resultNormPscBFM.bfm' | ignoreBFMLines ) \
            <( cat 'result.bfm'                    | ignoreBFMLines )
    local identical=$?
    if [ ! "$identical" -eq 0 ]; then
        if [ -f result.bfm ]; then
            colordiff --report-identical-files \
                <( cat '../tests/resultNormPscBFM.bfm' | ignoreBFMLines | hexdump -C ) \
                <( cat 'result.bfm'                    | ignoreBFMLines | hexdump -C ) | head -20
        else
            echo "File 'result.bfm' does not exist. It seems the simulation did not even finish and quit because of some problem."
        fi 2>&1 | tee -a "$logName-check.log"

        echo -e "\e[31mFiles are not identical, something is wrong, not bothering with benchmarks\e[0m" 1>&2
        return 1
    fi

    # print relevant timings
    echo "== checkSc $name =="
    'sed' -nr '/^t[A-Z].* = [0-9.]*s$/p' -- "$logName-check.log"
    if [ "$benchmark" -eq 0 ]; then return 0; fi

    # Get timings for kernels
    # nvprof run like this doesn't seem to add any measurable overhead => Call this in a loop to get statistics for tGpuLoop, tTaskLoop, ... instead of one-time values
    for (( i=0; i<5; ++i )); do
        $csrun nvprof "./$name" -i ../tests/inputPscBFM.bfm -e ../tests/resultNormPscBFM.seeds \
            -o result.bfm -g $iGpu -m 1000 -s 1000 2>&1 | tee -a "$logName-kernels.log"
    done

    # Get profling (time-consuming)
    # https://devblogs.nvidia.com/cuda-pro-tip-nvprof-your-handy-universal-gpu-profiler/
    $csrun nvprof --analysis-metrics --output-profile "$logName-profiling.nvvp" \
    "./$name" -i ../tests/inputPscBFM.bfm -e ../tests/resultNormPscBFM.seeds \
        -o result.bfm -g $iGpu -m 10 -s 10 2>&1 | tee "$logName-profiling.log"
    # note that this takes a looooong time: adding time command: 19.6s, vs. without: 2.3s and this is basically just the overhead with 100 steps it takes 2.6s and with 1000 6.2s => a+10*b=2.3, a+100*b=2.6 => 90*b=0.3s => 10 cycles shoul only take 0.0333s, but with nvprof take 19.6-2.3=17.3s which would mean that nvprof --anylsis-metrics incurs a 519x slowdown ...
)