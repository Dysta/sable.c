#include "easypap.h"

#include <stdbool.h>

static uint32_t *TABLE = NULL;

static volatile int changement;

static uint32_t max_grains;

#define table(i, j) TABLE[(i)*DIM + (j)]

#define RGB(r, v, b) (((r) << 24 | (v) << 16 | (b) << 8) | 255)

void sable_init()
{
    TABLE = calloc(DIM * DIM, sizeof(uint32_t));
}

void sable_finalize()
{
    free(TABLE);
}

///////////////////////////// Production d'une image
void sable_refresh_img()
{
    uint32_t max = 0;
    for (int i = 1; i < DIM - 1; i++)
        for (int j = 1; j < DIM - 1; j++)
        {
            int g = table(i, j);
            int r, v, b;
            r = v = b = 0;
            if (g == 1)
                v = 255;
            else if (g == 2)
                b = 255;
            else if (g == 3)
                r = 255;
            else if (g == 4)
                r = v = b = 255;
            else if (g > 4)
                r = b = 255 - (240 * ((double)g) / (double)max_grains);

            cur_img(i, j) = RGB(r, v, b);
            if (g > max)
                max = g;
        }

    max_grains = max;
}

///////////////////////////// Configurations initiales

static void sable_draw_4partout(void);

void sable_draw(char *param)
{
    // Call function ${kernel}_draw_${param}, or default function (second
    // parameter) if symbol not found
    hooks_draw_helper(param, sable_draw_4partout);
}

void sable_draw_4partout(void)
{
    max_grains = 8;
    for (int i = 1; i < DIM - 1; i++)
        for (int j = 1; j < DIM - 1; j++)
            cur_img(i, j) = table(i, j) = 4;
}

void sable_draw_DIM(void)
{
    max_grains = DIM;
    for (int i = DIM / 4; i < DIM - 1; i += DIM / 4)
        for (int j = DIM / 4; j < DIM - 1; j += DIM / 4)
            cur_img(i, j) = table(i, j) = i * j / 4;
}

void sable_draw_alea(void)
{
    max_grains = 5000;
    for (int i = 0; i<DIM>> 3; i++)
    {
        int i = 1 + random() % (DIM - 2);
        int j = 1 + random() % (DIM - 2);
        int grains = 1000 + (random() % (4000));
        cur_img(i, j) = table(i, j) = grains;
    }
}

///////////////////////////// Version séquentielle simple (seq)

static inline void compute_new_state(int y, int x)
{
    if (table(y, x) >= 4)
    {
        uint32_t div4 = table(y, x) / 4;
        table(y, x - 1) += div4;
        table(y, x + 1) += div4;
        table(y - 1, x) += div4;
        table(y + 1, x) += div4;
        table(y, x) %= 4;
        changement = 1;
    }
}

static void do_tile(int x, int y, int width, int height, int who)
{
    PRINT_DEBUG('c', "tuile [%d-%d][%d-%d] traitée\n", x, x + width - 1, y,
                y + height - 1);

    monitoring_start_tile(who);

    for (int i = y; i < y + height; i++)
        for (int j = x; j < x + width; j++)
        {
            compute_new_state(i, j);
        }
    monitoring_end_tile(x, y, width, height, who);
}

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned sable_compute_seq(unsigned nb_iter)
{

    for (unsigned it = 1; it <= nb_iter; it++)
    {
        changement = 0;
        // On traite toute l'image en un coup (oui, c'est une grosse tuile)
        do_tile(1, 1, DIM - 2, DIM - 2, 0);
        if (changement == 0)
            return it;
    }
    return 0;
}

///////////////////////////// Version séquentielle tuilée (tiled)

unsigned sable_compute_tiled(unsigned nb_iter)
{
    for (unsigned it = 1; it <= nb_iter; it++)
    {
        changement = 0;

        for (int y = 0; y < DIM; y += TILE_SIZE)
            for (int x = 0; x < DIM; x += TILE_SIZE)
                do_tile(x + (x == 0), y + (y == 0),
                        TILE_SIZE - ((x + TILE_SIZE == DIM) + (x == 0)),
                        TILE_SIZE - ((y + TILE_SIZE == DIM) + (y == 0)),
                        0 /* CPU id */);
        if (changement == 0)
            return it;
    }

    return 0;
}

/////////////////////// OpenCL
// static var for OCL version
static int *stable_tile;
static cl_mem ocl_changes, ocl_stable_tile;
#define TILE_N (SIZE/TILEX)

///// SYNCHRONIZED VERSION
// Suggested command line:
// ./run -k sable -o -v ocl_sync

void sable_init_ocl_sync(void)
{
    TABLE = calloc(DIM * DIM, sizeof(uint32_t));

    ocl_changes = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);
    if (!ocl_changes)
        exit_with_error("Failed to allocate ocl changes variable");
}

unsigned sable_invoke_ocl_sync(unsigned nb_iter)
{
    size_t global[2] = {SIZE, SIZE};  // global domain size for our calculation
    size_t local[2] = {TILEX, TILEY}; // local domain size for our calculation
    cl_int err;

    int current_changes;

    for (unsigned it = 1; it <= nb_iter; it++)
    {
        // on écrit 0 dans la détection des changements
        current_changes = 0;
        check(
            clEnqueueWriteBuffer(queue, ocl_changes, CL_TRUE, 0,
                                 sizeof(int), &current_changes, 0, NULL, NULL),
            "Failed to write to ocl_changes");
        // Set kernel arguments
        //
        err = 0;
        err |= clSetKernelArg(compute_kernel, 0, sizeof(cl_mem), &cur_buffer);
        err |= clSetKernelArg(compute_kernel, 1, sizeof(cl_mem), &next_buffer);
        err |= clSetKernelArg(compute_kernel, 2, sizeof(cl_mem), &ocl_changes);
        check(err, "Failed to set kernel arguments");

        err = clEnqueueNDRangeKernel(queue, compute_kernel, 2, NULL, global, local,
                                     0, NULL, NULL);
        check(err, "Failed to execute kernel");

        // On regarde si il y a eu des changements
        check(
            clEnqueueReadBuffer(queue, ocl_changes, CL_TRUE, 0,
                                sizeof(int), &current_changes, 0, NULL, NULL),
            "Failed to read in current_changes");

        // Swap buffers
        {
            cl_mem tmp = cur_buffer;
            cur_buffer = next_buffer;
            next_buffer = tmp;
        }

        if (current_changes == 0)
        {
            return it;
        }
    }

    return 0;
}

void sable_refresh_img_ocl_sync()
{
    cl_int err;
    err = clEnqueueReadBuffer(queue, cur_buffer, CL_TRUE, 0,
                              sizeof(uint32_t) * DIM * DIM, TABLE, 0, NULL, NULL);
    check(err, "Failed to read buffer from GPU");

    sable_refresh_img();
}

///// TILED VERSION
// Suggested command line:
// ./run -k sable -o -v ocl_tiled
void sable_init_ocl_tiled(void)
{
    TABLE       = calloc(DIM * DIM, sizeof(uint32_t));
    stable_tile = calloc(TILE_N * TILE_N, sizeof(int));

    //TODO: voir si on en a besoin
    ocl_changes = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);
    if (!ocl_changes)
        exit_with_error("Failed to allocate ocl changes variable");
    
    ocl_stable_tile = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * TILE_N * TILE_N, NULL, NULL);
    if (!ocl_stable_tile)
        exit_with_error("Failed to allocate stable tile buffer");

    check(
        clEnqueueWriteBuffer(queue, ocl_stable_tile, CL_TRUE, 0,
                                sizeof(int) * TILE_N * TILE_N, 
                                stable_tile, 0, NULL, NULL),
        "Failed to write to ocl_stable_change");
}

unsigned sable_invoke_ocl_tiled(unsigned nb_iter)
{
    size_t global[2] = {SIZE, SIZE};  // global domain size for our calculation
    size_t local[2] = {TILEX, TILEY}; // local domain size for our calculation
    cl_int err;
    
    for (unsigned it = 1; it <= nb_iter; it++)
    {
        // Set kernel arguments
        //
        err = 0;
        err |= clSetKernelArg(compute_kernel, 0, sizeof(cl_mem), &cur_buffer);
        err |= clSetKernelArg(compute_kernel, 1, sizeof(cl_mem), &next_buffer);
        err |= clSetKernelArg(compute_kernel, 2, sizeof(cl_mem), &ocl_stable_tile);
        check(err, "Failed to set kernel arguments");

        err = clEnqueueNDRangeKernel(queue, compute_kernel, 2, NULL, global, local,
                                    0, NULL, NULL);
        check(err, "Failed to execute kernel");

        // Swap buffers
        {
            cl_mem tmp = cur_buffer;
            cur_buffer = next_buffer;
            next_buffer = tmp;
        }

        err = clEnqueueReadBuffer(queue, ocl_stable_tile, CL_TRUE, 0,
                                    sizeof(int) * TILE_N * TILE_N, stable_tile, 0, NULL, NULL);

        for (int i = 0; i < TILE_N; i++) {
            for (int j = 0; j < TILE_N; j++) {
                printf("%d_%d[%d] ", i, j, stable_tile[j * TILE_N + i]);
            }
            printf("\n");
        }
        printf("\n");
    }

    return 0;
}

void sable_refresh_img_ocl_tiled()
{
    cl_int err;
    err = clEnqueueReadBuffer(queue, cur_buffer, CL_TRUE, 0,
                            sizeof(uint32_t) * DIM * DIM, TABLE, 0, NULL, NULL);
    check(err, "Failed to read buffer from GPU");

    sable_refresh_img();
}