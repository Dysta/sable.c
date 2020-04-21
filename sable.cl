#include "kernel/ocl/common.cl"

#define table(X, Y) in[(X)*DIM + (Y)]

__kernel void sable_ocl (__global unsigned *in, __global unsigned *out)
{
    int x = get_global_id (0);
    int y = get_global_id (1);
    
    unsigned tmp  = in[x * DIM + y];

    if (tmp > 4)
        tmp &= 3;

    tmp += in[x+1 * DIM + y] >> 2;
    tmp += in[x-1 * DIM + y] >> 2;
    tmp += in[x * DIM + y+1] >> 2;
    tmp += in[x * DIM + y-1] >> 2;
    
    out[x * DIM + y] = tmp;
}
