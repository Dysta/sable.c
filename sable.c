#include "easypap.h"

#include <omp.h>
#include <stdbool.h>

#if defined(ENABLE_VECTO) && (VEC_SIZE == 8)

static unsigned do_tile_vec(int x, int y, int width, int height);
static long unsigned int *buffer = NULL;

#else

#ifdef ENABLE_VECTO
#warning Only 256bit AVX (VEC_SIZE=8) vectorization is currently supported
#endif

#define do_tile_vec(x, y, w, h) do_tile((x), (y), (w), (h), (0))
#define compute_new_state_vec(i, j) compute_new_state(i, j)

#endif

static long unsigned int *TABLE = NULL;

static unsigned long int max_grains;

#define table(i, j) TABLE[(i)*DIM + (j)]

static inline long unsigned int* table_addr(int i, int j) {
    return &(TABLE[i * DIM + j]);
}

#define RGB(r, v, b) (((r) << 24 | (v) << 16 | (b) << 8) | 255)

void sable_init()
{
    // TABLE = calloc(DIM * DIM, sizeof(long unsigned int));
    TABLE = aligned_alloc(32, DIM * DIM * sizeof(long unsigned int));

    #if defined(ENABLE_VECTO) && (VEC_SIZE == 8)
    buffer = _mm_malloc(DIM * DIM * sizeof(long unsigned int), 32);
    #endif
}

void sable_finalize()
{
    free(TABLE);

    #if defined(ENABLE_VECTO) && (VEC_SIZE == 8)
    _mm_free(buffer);
    #endif
}

///////////////////////////// Production d'une image
void sable_refresh_img()
{
    unsigned long int max = 0;
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
            table(i, j) = 4;
}

void sable_draw_DIM(void)
{
    max_grains = DIM;
    for (int i = DIM / 4; i < DIM - 1; i += DIM / 4)
        for (int j = DIM / 4; j < DIM - 1; j += DIM / 4)
            table(i, j) = i * j / 4;
}

void sable_draw_alea(void)
{
    max_grains = 5000;
    for (int i = 0; i<DIM>> 3; i++)
    {
        table(1 + random() % (DIM - 2), 1 + random() % (DIM - 2)) =
            1000 + (random() % (4000));
    }
}

///////////////////////////// Version séquentielle simple (seq)
static inline unsigned compute_new_state(int y, int x)
{
    if (table(y, x) >= 4)
    {
        unsigned long int div4 = table(y, x) / 4;
        table(y, x - 1) += div4;
        table(y, x + 1) += div4;
        table(y - 1, x) += div4;
        table(y + 1, x) += div4;
        table(y, x) %= 4;
        return 1;
    }
    return 0;
}


static unsigned do_tile(int x, int y, int width, int height, int who)
{
    PRINT_DEBUG('c', "tuile [%d-%d][%d-%d] traitée\n", x, x + width - 1, y,
                y + height - 1);

    monitoring_start_tile(who);
    unsigned changes = 0;
    for (int i = y; i < y + height; i++)
        for (int j = x; j < x + width; j++)
        {   
            changes += compute_new_state(i, j);
        }
    monitoring_end_tile(x, y, width, height, who);

    return changes;
}

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned sable_compute_seq(unsigned nb_iter)
{
    unsigned changement = 0;
    for (unsigned it = 1; it <= nb_iter; it++)
    {
        // On traite toute l'image en un coup (oui, c'est une grosse tuile)
        changement += do_tile(1, 1, DIM - 2, DIM - 2, 0);
        if (changement == 0)
            return it;
        changement = 0;
    }
    return 0;
}

///////////////////////////// Version séquentielle tuilée (tiled)
unsigned sable_compute_tiled(unsigned nb_iter)
{
    for (unsigned it = 1; it <= nb_iter; it++)
    {
        unsigned changement = 0;

        for (int y = 0; y < DIM; y += TILE_SIZE)
            for (int x = 0; x < DIM; x += TILE_SIZE)
                changement += do_tile(x + (x == 0), y + (y == 0),
                                TILE_SIZE - ((x + TILE_SIZE == DIM) + (x == 0)),
                                TILE_SIZE - ((y + TILE_SIZE == DIM) + (y == 0)),
                                0 /* CPU id */);
        if (changement == 0)
            return it;
        
        changement = 0;
    }

    return 0;
}

///////////////////////////// Version ompfor_bar (tiled)
// FIXME: cause des bugs de segmentation
unsigned sable_compute_ompfor_n(unsigned nb_iter)
{
    for (unsigned it = 1; it <= nb_iter; it++)
    {
        unsigned changes = 0;
        const int max_tile_idx = DIM / TILE_SIZE + (DIM % TILE_SIZE > 0);

        #pragma omp parallel for reduction(+:changes)
        for (int y = 0; y < max_tile_idx; y+=2)
        {
            const int x_start   = 1;
            const int y_start   = y * TILE_SIZE + (y == 0);
            const int width     = DIM - 2;
            const int height = (y == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;

            changes += do_tile(x_start, y_start, width, height, omp_get_thread_num());
        }

        // #pragma omp barrier

        #pragma omp parallel for reduction(+:changes)
        for (int y = 1; y < max_tile_idx; y+=2)
        {
            const int x_start   = 1;
            const int y_start   = y * TILE_SIZE;
            const int width     = DIM - 2;
            const int height = (y == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;

            changes += do_tile(x_start, y_start, width, height, omp_get_thread_num());
        }

        if (changes == 0)
            return it;

        changes = 0;
    }

    return 0;
}

unsigned sable_compute_ompfor_pair_impair(unsigned nb_iter)
{
    for (unsigned it = 1; it <= nb_iter; it++)
    {
        unsigned changes = 0;

        #pragma omp parallel for reduction(+:changes)
        for (int y = 0; y < DIM; y+=TILE_SIZE)
        {
            if (y % 2 == 0)
                changes += do_tile(1, y + (y == 0),
                            DIM - 2 ,
                            TILE_SIZE - ((y + TILE_SIZE == DIM) + (y == 0)),
                            omp_get_thread_num());
        }

        // #pragma omp barrier

        #pragma omp parallel for reduction(+:changes)
        for (int y = 0; y < DIM; y+=TILE_SIZE)
        {
            if (y % 2 != 0)
                changes += do_tile(1, y + (y == 0),
                            DIM - 2 ,
                            TILE_SIZE - ((y + TILE_SIZE == DIM) + (y == 0)),
                            omp_get_thread_num());
        }

        if (changes == 0)
            return it;

        changes = 0;
    }

    return 0;
}

///////////////////////////// Version vec (tiled) (not working)
unsigned sable_compute_vec(unsigned nb_iter)
{
    unsigned changement = 0;
    for (unsigned it = 1; it <= nb_iter; it++)
    {
        // On traite toute l'image en un coup (oui, c'est une grosse tuile)
        changement += do_tile_vec(1, 1, DIM - 2, DIM - 2);
        if (changement == 0)
            return it;
        changement = 0;
    }
    return 0;
}

#if defined(ENABLE_VECTO) && (VEC_SIZE == 8)
#include <immintrin.h>

static unsigned compute_new_state_vec(int y, int x)
{
    PRINT_DEBUG('c', "compute new state en [y x]-[%d %d]\n", y, x);
    /*
    if (table(y, x) >= 4)
    {
        unsigned long int div4 = table(y, x) / 4; -> table(y, x) >> 2
        table(y, x - 1) += div4;
        table(y, x + 1) += div4;
        table(y - 1, x) += div4;
        table(y + 1, x) += div4;
        table(y, x) %= 4; -> &= 3
        return 1;
    }
    return 0;
    */
    // création d'un vecteur contenant table(y, x)
    __m256i vecTable = _mm256_loadu_si256((const void*) table_addr(y, x));
    // création des vecteurs pour comparaisons et opérations
    __m256i vecThree = _mm256_set1_epi32(3);

    // compare vecTable[x] > 3, 0xFF si vrai, 0 sinon
    // reviens à comparer vecTable[x] >= 4
    __m256i mask = _mm256_cmpgt_epi32(vecTable, vecThree);

    // Si il y a que des 0 dans le mask, on s'arrête la
    if (_mm256_testz_si256(mask, mask))
        return 0;

    // div4 = table(y, x) / 4
    // soit div4[x] contient table(y, x) / 4, 0 autrement (pour ne pas changer l'addition)
    __m256i div4 = _mm256_srli_epi32(vecTable, 2);

    // création des vecteurs contenant table(y - 1, x) et table(y + 1, x)
    __m256i vecTable_XInf = _mm256_loadu_si256((const void*) table_addr(y-1,x));
    __m256i vecTable_XSup = _mm256_loadu_si256((const void*) table_addr(y+1,x));
    __m256i vecTable_YInf = _mm256_loadu_si256((const void*) table_addr(y,x-1));
    __m256i vecTable_YSup = _mm256_loadu_si256((const void*) table_addr(y,x+1));

    // table(y - 1, x) += div4
    vecTable_XInf = _mm256_add_epi32(vecTable_XInf, div4);
    // table(y + 1, x) += div4
    vecTable_XSup = _mm256_add_epi32(vecTable_XSup, div4);
    // table(y, x - 1) += div4
    vecTable_YInf = _mm256_add_epi32(vecTable_YInf, div4);
    // table(y, x + 1) += div4
    vecTable_YSup = _mm256_add_epi32(vecTable_YSup, div4);

    _mm256_storeu_si256((__m256i*) table_addr(y-1, x), vecTable_XInf);
    _mm256_storeu_si256((__m256i*) table_addr(y+1, x), vecTable_XSup);
    _mm256_storeu_si256((__m256i*) table_addr(y, x-1), vecTable_YInf);
    _mm256_storeu_si256((__m256i*) table_addr(y, x+1), vecTable_YSup);

    // table(y, x) %= 4 -> &= 3
    __m256i modulus_data = _mm256_and_si256(vecTable, vecThree);
    _mm256_storeu_si256((__m256i*) table_addr(y, x), modulus_data);

    return 1;
}

static unsigned do_tile_vec(int x, int y, int width, int height)
{

    PRINT_DEBUG('c', "tuile [%d-%d][%d-%d] traitée\n", x, x + width - 1, y,
                y + height - 1);

    unsigned changement = 0;
    for (int i = y; i < y + height; i++) {
        for (int j = x; j < x + width; j += VEC_SIZE)
            changement += compute_new_state_vec(i, j);
    }
    return changement;
}

#endif