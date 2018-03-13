#include "Hash.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include "PCG.h"

uint32_t hashLEMONADE( uint32_t a ) {
 a = ( a + 0x7ed55d16 ) + ( a << 12 );
 a = ( a ^ 0xc761c23c ) ^ ( a >> 19 );
 a = ( a + 0x165667b1 ) + ( a << 5  );
 a = ( a + 0xd3a2646c ) ^ ( a << 9  );
 a = ( a + 0xfd7046c5 ) + ( a << 3  );
 a = ( a ^ 0xb55a4f09 ) ^ ( a >> 16 );
 return a;
}

using namespace std;

template<typename RNG>
void kernel( const unsigned int seed,
             const unsigned int iteration,
             typename RNG::global_state_type*global_state_array,
             const unsigned int Nmonomers)
    {
    for(unsigned int mono=0; mono < Nmonomers; mono++)
        {
        RNG rng;

        if( RNG::needs_global_state() )
            rng.set_global_state(global_state_array+mono);
        if( RNG::needs_iteration() )
            rng.set_iteration(iteration);
        if( RNG::needs_subsequence() )
            rng.set_subsequence(mono);
        if( RNG::needs_seed() )
            rng.set_seed( seed );

        cout<<rng.rng32()<<" "<<rng.rng32()<<" "<<rng.rng32() <<endl;
        }
    //cout<<endl<<endl<<endl;
    }



int main(int argc,char*argv[])
    {

    const unsigned int Nsteps = 12000;

    const unsigned int seed = 42;

    const unsigned int Nmonomers = 123;

    std::vector<PCG::State> state_vector(Nmonomers);
    for(unsigned int i=0; i < Nmonomers; i++)
        state_vector[i] = PCG::State(seed,i);
    PCG::global_state_type* global_state_array = state_vector.data();

    for(unsigned int global_step=0; global_step < Nsteps; global_step++)
        kernel<PCG>(seed,global_step,global_state_array,Nmonomers);

    return 0;
    }
