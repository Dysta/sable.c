#include "kernel/ocl/common.cl"

__kernel void sable_ocl_sync (__global unsigned *in, __global unsigned *out, __global int* changes)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    
    out[y * DIM + x] = in[y * DIM + x] % 4;

    // Update gauche/droite
    if (x + 1 <= DIM - 1)
        out[y * DIM + x] += in[y * DIM + (x + 1)] / 4;
    if (x - 1 >= 0)
        out[y * DIM + x] += in[y * DIM + (x - 1)] / 4;

    // Update haut/bas
    if (y + 1 <= DIM - 1)
        out[y * DIM + x] += in[(y + 1) * DIM + x] / 4;
    if (y - 1 >= 0)
        out[y * DIM + x] += in[(y - 1) * DIM + x] / 4;

    // barrier(CLK_LOCAL_MEM_FENCE);

    // Si on a fait un changement on met changes à 1
    if (*changes == 0 && out[y * DIM + x] != in[y * DIM + x])
        *changes = 1;
}

#define TILE_N (SIZE/TILEX)

__kernel void sable_ocl_tiled (__global unsigned *in, __global unsigned *out, 
                                __global int* in_stable_tile, __global int* out_stable_tile, 
                                __global int* changes)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int xtile = get_group_id(0);
    const int ytile = get_group_id(1);

    // Si la tile est stable, on passe
    if (in_stable_tile[ytile * TILE_N + xtile] == 0) {
        out[y * DIM + x] = in[y * DIM + x] % 4;

        // Update gauche/droite
        if (x + 1 <= DIM - 1)
            out[y * DIM + x] += in[y * DIM + (x + 1)] / 4;
        if (x - 1 >= 0)
            out[y * DIM + x] += in[y * DIM + (x - 1)] / 4;

        // Update haut/bas
        if (y + 1 <= DIM - 1)
            out[y * DIM + x] += in[(y + 1) * DIM + x] / 4;
        if (y - 1 >= 0)
            out[y * DIM + x] += in[(y - 1) * DIM + x] / 4;
        
        out_stable_tile[ytile * TILE_N + xtile] = 0;
        in_stable_tile[ytile * TILE_N + xtile] = 1;
    }

    if (*changes == 0 && out[y * DIM + x] != in[y * DIM + x])
        *changes = 1;
}

// DO NOT MODIFY: this kernel updates the OpenGL texture buffer
// This is a sable-specific version (generic version is defined in common.cl)
__kernel void sable_update_texture (__global unsigned *cur, __write_only image2d_t tex)
{
    int y = get_global_id (1);
    int x = get_global_id (0);
    int2 pos = (int2)(x, y);
    unsigned c = cur [y * DIM + x];
    unsigned r = 0, v = 0, b = 0;

    if (c == 1)
        v = 255;
    else if (c == 2)
        b = 255;
    else if (c == 3)
        r = 255;
    else if (c == 4)
        r = v = b = 255;
    else if (c > 4)
        r = v = b = (2 * c);

    c = rgba(r, v, b, 0xFF);

    write_imagef (tex, pos, color_scatter (c));
}
