#include "pscbfm/cudacommon.hpp"

int main( void )
{
    cudaDeviceProp * pGpus = NULL;
    int              nGpus = 0   ;
    getCudaDeviceProperties( &pGpus, &nGpus, true );
    return 0;
}
