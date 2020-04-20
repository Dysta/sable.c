#include "easypap.h"

#include <omp.h>
#include <stdbool.h>
#include <assert.h>

#if defined(ENABLE_VECTO) && (VEC_SIZE == 8)

static unsigned do_tile_vec(int x, int y, int width, int height, int who);
static uint64_t *TABLE_vec = NULL;

#define table_vec(X, Y) TABLE_vec[(X)*(DIM + 32) + (Y) + DIM - 1]

static inline uint64_t* table_addr(const int i, const int j) {
    return &(table_vec(i, j));
}

#else

#ifdef ENABLE_VECTO
#warning Only 256bit AVX (VEC_SIZE=8) vectorization is currently supported
#endif

#define do_tile_vec(x, y, w, h, who) do_tile((x), (y), (w), (h), (who))
#define compute_new_state_vec(i, j) compute_new_state(i, j)

#endif

static uint64_t *TABLE = NULL;

static uint64_t max_grains;

#define table(X, Y) TABLE[(X) * DIM + (Y)]

#define RGB(r, v, b) (((r) << 24 | (v) << 16 | (b) << 8) | 255)

void sable_init()
{
    TABLE = calloc(DIM * DIM, sizeof(uint64_t));

    #if defined(ENABLE_VECTO) && (VEC_SIZE == 8)
    // trying to allign the table addr to a multiple of 32 (to use load instead of loadu)
    TABLE_vec = aligned_alloc(32, (DIM + 32) * (DIM + 1) * sizeof(uint64_t));
    memset(TABLE_vec, 0, (DIM + 32) * (DIM + 1) * sizeof(uint64_t));
    #endif
}

void sable_finalize()
{
    free(TABLE);

    #if defined(ENABLE_VECTO) && (VEC_SIZE == 8)
    free(TABLE_vec);
    #endif
}

///////////////////////////// Production d'une image
void sable_refresh_img()
{
    uint64_t max = 0;
    for (int i = 1; i < DIM - 1; i++)
        for (int j = 1; j < DIM - 1; j++)
        {
            //FIXME: ne pas oublier de commenter/décommenter
            // int g = table(i, j);
            int g = table_vec(i, j);

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
        for (int j = 1; j < DIM - 1; j++) {
            table(i, j) = 4;
            #if defined(ENABLE_VECTO) && (VEC_SIZE == 8)
            table_vec(i, j) = 4;
            #endif
        }
}

void sable_draw_DIM(void)
{
    max_grains = DIM;
    for (int i = DIM / 4; i < DIM - 1; i += DIM / 4)
        for (int j = DIM / 4; j < DIM - 1; j += DIM / 4) {
            table(i, j) = i * j / 4;
            #if defined(ENABLE_VECTO) && (VEC_SIZE == 8)
            table_vec(i, j) = i * j / 4;
            #endif
        }
}

void sable_draw_alea(void)
{
    max_grains = 5000;
    for (int i = 0; i<DIM>> 3; i++)
    {
        const unsigned x = 1 + random() % (DIM - 2);
        const unsigned y = 1 + random() % (DIM - 2);
        const unsigned g = 1000 + (random() % (4000));
        table(x, y) = g;
        #if defined(ENABLE_VECTO) && (VEC_SIZE == 8)
        table_vec(x, y) = g;
        #endif
    }
}

///////////////////////////// Version séquentielle simple (seq)
static inline unsigned compute_new_state(int y, int x)
{
    if (table(y, x) >= 4)
    {
        unsigned long int div4 = table(y, x) >> 2;
        table(y, x - 1) += div4;
        table(y, x + 1) += div4;
        table(y - 1, x) += div4;
        table(y + 1, x) += div4;
        table(y, x) &= 3;
        return 1;
    }
    return 0;
}


static unsigned do_tile(int x, int y, int width, int height, int who)
{
    PRINT_DEBUG('c', "tuile [%d-%d][%d-%d] traitée\n", x, x + width - 1, y, y + height - 1);

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
unsigned sable_compute_ompfor(unsigned nb_iter)
{
    for (unsigned it = 1; it <= nb_iter; it++)
    {
        unsigned changes = 0;
        const int max_tile_idx = (DIM / TILE_SIZE + (DIM % TILE_SIZE > 0));
        
        #pragma omp parallel for reduction(+:changes) schedule(runtime)
        for (int y = 0; y < max_tile_idx; y+=2)
        {
            const int x_start   = 1;
            const int y_start   = y * TILE_SIZE + (y == 0);
            const int width     = DIM - 2;
            const int height = (y == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE - (y == max_tile_idx - 1);
            
            changes += do_tile(x_start, y_start, width, height, omp_get_thread_num());
        }

        #pragma omp parallel for reduction(+:changes) schedule(runtime)
        for (int y = 1; y < max_tile_idx; y+=2)
        {
            const int x_start   = 1;
            const int y_start   = y * TILE_SIZE;
            const int width     = DIM - 2;
            const int height = (y == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE - (y == max_tile_idx - 1);
            
            changes += do_tile(x_start, y_start, width, height, omp_get_thread_num());
        }

        if (changes == 0)
            return it;

        changes = 0;
    }

    return 0;
}

///////////////////////////// Version vec (tiled) (not working)
unsigned sable_compute_vec_ompfor(unsigned nb_iter)
{
    if (DIM % 8 != 0) {
        fprintf(stderr, "Error: DIM (%d) must be a multiple of 8 for vectorial compute\n", DIM);
        assert(DIM % 8 == 0);
    }

    unsigned changes = 0;
    for (unsigned it = 1; it <= nb_iter; it++)
    {
        const int max_tile_idx = (DIM / TILE_SIZE + (DIM % TILE_SIZE > 0));
        
        #pragma omp parallel for reduction(+:changes) schedule(runtime)
        for (int y = 0; y < max_tile_idx; y+=2)
        {
            const int x_start   = 1;
            const int y_start   = y * TILE_SIZE + (y == 0);
            const int width     = DIM - 2;
            const int height = (y == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE - (y == max_tile_idx - 1);
            
            changes += do_tile_vec(x_start, y_start, width, height, omp_get_thread_num());
        }

        #pragma omp parallel for reduction(+:changes) schedule(runtime)
        for (int y = 1; y < max_tile_idx; y+=2)
        {
            const int x_start   = 1;
            const int y_start   = y * TILE_SIZE;
            const int width     = DIM - 2;
            const int height = (y == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE - (y == max_tile_idx - 1);
            
            changes += do_tile_vec(x_start, y_start, width, height, omp_get_thread_num());
        }

        if (changes == 0)
            return it;

        changes = 0;
    }
    return 0;
}

unsigned sable_compute_vec(unsigned nb_iter)
{
    if (DIM % 8 != 0) {
        fprintf(stderr, "Error: DIM (%d) must be a multiple of 8 for vectorial compute\n", DIM);
        assert(DIM % 8 == 0);
    }

    unsigned changement = 0;
    for (unsigned it = 1; it <= nb_iter; it++)
    {
        // On traite toute l'image en un coup (oui, c'est une grosse tuile)
        changement += do_tile_vec(1, 1, DIM - 2, DIM - 2, 0);
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
    void* addr_center = (void*) table_addr(y, x);
    void* addr_up     = (void*) table_addr(y + 1, x);
    void* addr_down   = (void*) table_addr(y - 1, x);
    void* addr_left   = (void*) table_addr(y, x - 1);
    void* addr_right  = (void*) table_addr(y, x + 1);

    // création d'un vecteur contenant table(y, x)
    __m256i vecTable_center = _mm256_load_si256(addr_center);
    // création des vecteurs pour comparaisons et opérations
    __m256i vecThree = _mm256_set_epi32(3, 3, 3, 3, 3, 3, 3, 3);

    // compare vecTable[x] > 3, 0xFF si vrai, 0 sinon
    // reviens à comparer vecTable[x] >= 4
    __m256i mask = _mm256_cmpgt_epi32(vecTable_center, vecThree);

    // Si il y a que des 0 dans le mask, on s'arrête la
    if (_mm256_testz_si256(mask, mask))
        return 0;

    // div4 = table(y, x) / 4
    // soit div4[x] contient table(y, x) / 4, 0 autrement (pour ne pas changer l'addition)
    __m256i div4 = _mm256_srli_epi32(vecTable_center, 2);

    // table(y, x) %= 4 -> &= 3
    __m256i modulus_data = _mm256_and_si256(vecTable_center, vecThree);
    _mm256_store_si256(addr_center, modulus_data);

    // création des vecteurs contenant table(y - 1, x)
    // table(y - 1, x) += div4
    // puis store dans l'image à la même place
    __m256i vecTable_left = _mm256_loadu_si256(addr_left);
    vecTable_left = _mm256_add_epi32(vecTable_left, div4);
    _mm256_storeu_si256(addr_left, vecTable_left);

    __m256i vecTable_right = _mm256_loadu_si256(addr_right);
    vecTable_right = _mm256_add_epi32(vecTable_right, div4);
    _mm256_storeu_si256(addr_right, vecTable_right);

    __m256i vecTable_down = _mm256_load_si256(addr_down);
    vecTable_down = _mm256_add_epi32(vecTable_down, div4);
    _mm256_store_si256(addr_down, vecTable_down);

    __m256i vecTable_up = _mm256_load_si256(addr_up);
    vecTable_up = _mm256_add_epi32(vecTable_up, div4);
    _mm256_store_si256(addr_up, vecTable_up);

    return 1;
}

static unsigned do_tile_vec(int x, int y, int width, int height, int who)
{

    PRINT_DEBUG('c', "tuile [%d-%d][%d-%d] traitée\n", x, x + width - 1, y,
                y + height - 1);

    monitoring_start_tile(who);
    unsigned changement = 0;
    for (int i = y; i < y + height; i++) {
        for (int j = x; j < x + width; j += VEC_SIZE/2)
            changement += compute_new_state_vec(i, j);
    }
    monitoring_end_tile(x, y, width, height, who);

    return changement;
}

#endif