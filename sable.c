#include "easypap.h"

#include <omp.h>
#include <stdbool.h>

#if defined(ENABLE_VECTO) && (VEC_SIZE == 8)

static void do_tile_vec(int x, int y, int width, int height);

#else

#ifdef ENABLE_VECTO
#warning Only 256bit AVX (VEC_SIZE=8) vectorization is currently supported
#endif

#define do_tile_vec(x, y, w, h) do_tile((x), (y), (w), (h), (0))

#endif

static long unsigned int *TABLE = NULL;

static volatile int changement;

static unsigned long int max_grains;

#define table(i, j) TABLE[(i)*DIM + (j)]

#define RGB(r, v, b) (((r) << 24 | (v) << 16 | (b) << 8) | 255)

void sable_init()
{
    TABLE = calloc(DIM * DIM, sizeof(long unsigned int));
}

void sable_finalize()
{
    free(TABLE);
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

static inline void compute_new_state(int y, int x)
{
    if (table(y, x) >= 4)
    {
        unsigned long int div4 = table(y, x) / 4;
        table(y, x - 1) += div4;
        table(y, x + 1) += div4;
        table(y - 1, x) += div4;
        table(y + 1, x) += div4;
        table(y, x) %= 4;
        changement = 1;
    }
}

static inline void compute_new_state_omp(int y, int x)
{
    if (table(y, x) < 4) return;

    if (table(y, x) >= 4)
    {
        unsigned long int div4;
        div4 = table(y, x) / 4;
        table(y, x) %= 4;
        #pragma omp critical
        {
            table(y, x - 1) += div4;
            table(y, x + 1) += div4;
            table(y - 1, x) += div4;
            table(y + 1, x) += div4;
        }
        changement = 1;
    }
}

static inline void compute_new_state_omp_dwleft(int y, int x)
{
    if (table(y, x) < 4) return;

    if (table(y, x) >= 4)
    {
        unsigned long int div4;
        div4 = table(y, x) / 4;
        table(y, x) %= 4;
        #pragma omp critical
        {
            table(y, x + 1) += div4;
            table(y - 1, x) += div4;
        }
        table(y, x - 1) += div4;
        table(y + 1, x) += div4;
        changement = 1;
    }
}

static inline void compute_new_state_omp_dwright(int y, int x)
{
    if (table(y, x) < 4) return;

    if (table(y, x) >= 4)
    {
        unsigned long int div4;
        div4 = table(y, x) / 4;
        table(y, x) %= 4;
        #pragma omp critical
        {
            table(y, x + 1) += div4;
            table(y + 1, x) += div4;
        }
        table(y, x - 1) += div4;
        table(y - 1, x) += div4;
        changement = 1;
    }
}

static inline void compute_new_state_omp_upleft(int y, int x)
{
    if (table(y, x) < 4) return;

    if (table(y, x) >= 4)
    {
        unsigned long int div4;
        div4 = table(y, x) / 4;
        table(y, x) %= 4;
        #pragma omp critical
        {
            table(y, x + 1) += div4;
            table(y - 1, x) += div4;
        }
        table(y, x - 1) += div4;
        table(y + 1, x) += div4;
        changement = 1;
    }
}

static inline void compute_new_state_omp_upright(int y, int x)
{
    if (table(y, x) < 4) return;

    if (table(y, x) >= 4)
    {
        unsigned long int div4;
        div4 = table(y, x) / 4;
        table(y, x) %= 4;
        #pragma omp critical
        {
            table(y, x - 1) += div4;
            table(y + 1, x) += div4;
        }
        table(y, x + 1) += div4;
        table(y - 1, x) += div4;
        changement = 1;
    }
}

static inline void compute_new_state_omp_up(int y, int x)
{
    if (table(y, x) < 4) return;

    if (table(y, x) >= 4)
    {
        unsigned long int div4;
        div4 = table(y, x) / 4;
        table(y, x) %= 4;
        #pragma omp critical
        {
            table(y, x - 1) += div4;
        }
        table(y + 1, x) += div4;
        table(y, x + 1) += div4;
        table(y - 1, x) += div4;
        changement = 1;
    }
}

static inline void compute_new_state_omp_down(int y, int x)
{
    if (table(y, x) < 4) return;

    if (table(y, x) >= 4)
    {
        unsigned long int div4;
        div4 = table(y, x) / 4;
        table(y, x) %= 4;
        #pragma omp critical
        {
            table(y + 1, x) += div4;
        }
        table(y - 1, x) += div4;
        table(y, x - 1) += div4;
        table(y, x + 1) += div4;
        changement = 1;
    }
}

static inline void compute_new_state_omp_left(int y, int x)
{
    if (table(y, x) < 4) return;

    if (table(y, x) >= 4)
    {
        unsigned long int div4;
        div4 = table(y, x) / 4;
        table(y, x) %= 4;
        #pragma omp critical
        {
            table(y, x - 1) += div4;
        }
        table(y - 1, x) += div4;
        table(y + 1, x) += div4;
        table(y, x + 1) += div4;
        changement = 1;
    }
}

static inline void compute_new_state_omp_right(int y, int x)
{
    if (table(y, x) < 4) return;

    if (table(y, x) >= 4)
    {
        unsigned long int div4;
        div4 = table(y, x) / 4;
        table(y, x) %= 4;
        #pragma omp critical
        {
            table(y, x + 1) += div4;
        }
        table(y - 1, x) += div4;
        table(y + 1, x) += div4;
        table(y, x - 1) += div4;
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



static void do_tile_omp(int x, int y, int width, int height, int who)
{
    PRINT_DEBUG('c', "tuile [%d-%d][%d-%d] traitée\n", x, x + width - 1, y,
                y + height - 1);

    monitoring_start_tile(who);

    for (int i = y; i < y + height; i++)
        for (int j = x; j < x + width; j++)
        {
            if (i == y && j == x) // en haut à gauche
                compute_new_state_omp(i, j);
                // compute_new_state_omp_upleft(i, j);
            else if (i == y && j == x + width - 1) // en haut à droite
                compute_new_state_omp(i, j);
                // compute_new_state_omp_upright(i, j);
            else if (i == y + height - 1 && j == x) // en bas à gauche
                compute_new_state_omp(i, j);
                // compute_new_state_omp_dwleft(i, j);
            else if (i == y + height - 1 && j == x + width - 1) // en bas à droite
                compute_new_state_omp(i, j);
                // compute_new_state_omp_dwright(i, j);
            // else if (i == y) // ligne haut d'une tuile
            //     compute_new_state_omp_up(i, j);
            // else if (i == y + height - 1) // en bas d'une tuile
            //     compute_new_state_omp_down(i, j);
            // else if (j == x) // ligne gauche
            //     compute_new_state_omp_left(i, j);
            // else if (j == x + width - 1) // ligne droite
            //     compute_new_state_omp_right(i, j);
            else // au centre
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

///////////////////////////// Version ompfor (tiled) (not working)
// FIXME: améliorer car plus lent que version seq
unsigned sable_compute_ompfor(unsigned nb_iter)
{
    for (unsigned it = 1; it <= nb_iter; it++)
    {
        changement = 0;

        #pragma omp parallel for collapse(2) schedule(static)
        for (int y = 0; y < DIM; y += TILE_SIZE)
        {
            for (int x = 0; x < DIM; x += TILE_SIZE)
            {
                do_tile_omp(x + (x == 0), y + (y == 0),
                            TILE_SIZE - ((x + TILE_SIZE == DIM) + (x == 0)),
                            TILE_SIZE - ((y + TILE_SIZE == DIM) + (y == 0)),
                            omp_get_thread_num() /* CPU id */);
            }
        }
        if (changement == 0)
            return it;
    }

    return 0;
}

///////////////////////////// Version vec (tiled) (not working)
unsigned sable_compute_vec(unsigned nb_iter)
{

    for (unsigned it = 1; it <= nb_iter; it++)
    {
        changement = 0;
        // On traite toute l'image en un coup (oui, c'est une grosse tuile)
        do_tile_vec(1, 1, DIM - 2, DIM - 2);
        if (changement == 0)
            return it;
    }
    return 0;
}

#if defined(ENABLE_VECTO) && (VEC_SIZE == 8)
#include <immintrin.h>

static void compute_new_state_vec(int y, int x)
{
    /*
    if (table(y x) >= 4) {
        unsigned long int div4;
        div4 = table(y, x) / 4;
        table(y, x - 1) = div4;
        table(y, x + 1) = div4;
        table(y - 1, x) = div4;
        table(x + 1, x) = div4;
        table(y , x) %= 4;
        chagement = 1;
    }
    */

    // création d'un vecteur contenant table(y, x)
    __m256i vecTable = _mm256_set_epi32(
        table(y, x + 7),
        table(y, x + 6),
        table(y, x + 5),
        table(y, x + 4),
        table(y, x + 3),
        table(y, x + 2),
        table(y, x + 1),
        table(y, x));
    // création des vecteurs pour comparaisons et opérations
    __m256i vecThree = _mm256_set1_epi32(3);
    __m256i vecZero = _mm256_set1_epi32(0);

    // compare vecTable[x] > 3, 0xFF si vrai, 0 sinon
    // reviens à comparer vecTable[x] >= 4
    __m256i mask = _mm256_cmpgt_epi32(vecTable, vecThree);

    // Si il y a que des 0 dans le mask, on s'arrête la
    if (_mm256_testz_si256(mask, mask))
        return;

    // création du vecteur contenant des 4 pour la division
    __m256 vecFour = _mm256_set1_ps(4.0);

    // on créer un vecteur qui contient table(y, x) si >= 4, 0 autrement
    __m256 vecTableModif = _mm256_blendv_ps(
        _mm256_cvtepi32_ps(vecZero),
        _mm256_cvtepi32_ps(vecTable),
        _mm256_cvtepi32_ps(mask));

    // div4 = table(y, x) / 4 puis une conversion du resultat float -> int
    // soit div4[x] contient table(y, x) / 4 si mask, 0 autrement (pour ne pas changer l'addition)
    __m256i div4 = _mm256_cvtps_epi32(_mm256_div_ps(vecTableModif, vecFour));

    // création des vecteurs contenant table(y - 1, x) et table(y + 1, x)
    __m256i vecTable_XInf = _mm256_set_epi32(
        table(y - 1, x + 7),
        table(y - 1, x + 6),
        table(y - 1, x + 5),
        table(y - 1, x + 4),
        table(y - 1, x + 3),
        table(y - 1, x + 2),
        table(y - 1, x + 1),
        table(y - 1, x));
    __m256i vecTable_XSup = _mm256_set_epi32(
        table(y + 1, x + 7),
        table(y + 1, x + 6),
        table(y + 1, x + 5),
        table(y + 1, x + 4),
        table(y + 1, x + 3),
        table(y + 1, x + 2),
        table(y + 1, x + 1),
        table(y + 1, x));

    // table(y - 1, x) += div4
    vecTable_XInf = _mm256_add_epi32(vecTable_XInf, div4);
    // table(y + 1, x) += div4
    vecTable_XSup = _mm256_add_epi32(vecTable_XSup, div4);

    table(y - 1, x) = _mm256_extract_epi32(vecTable_XInf, 0);
    table(y - 1, x + 1) = _mm256_extract_epi32(vecTable_XInf, 1);
    table(y - 1, x + 2) = _mm256_extract_epi32(vecTable_XInf, 2);
    table(y - 1, x + 3) = _mm256_extract_epi32(vecTable_XInf, 3);
    table(y - 1, x + 4) = _mm256_extract_epi32(vecTable_XInf, 4);
    table(y - 1, x + 5) = _mm256_extract_epi32(vecTable_XInf, 5);
    table(y - 1, x + 6) = _mm256_extract_epi32(vecTable_XInf, 6);
    table(y - 1, x + 7) = _mm256_extract_epi32(vecTable_XInf, 7);

    table(y + 1, x) = _mm256_extract_epi32(vecTable_XSup, 0);
    table(y + 1, x + 1) = _mm256_extract_epi32(vecTable_XSup, 1);
    table(y + 1, x + 2) = _mm256_extract_epi32(vecTable_XSup, 2);
    table(y + 1, x + 3) = _mm256_extract_epi32(vecTable_XSup, 3);
    table(y + 1, x + 4) = _mm256_extract_epi32(vecTable_XSup, 4);
    table(y + 1, x + 5) = _mm256_extract_epi32(vecTable_XSup, 5);
    table(y + 1, x + 6) = _mm256_extract_epi32(vecTable_XSup, 6);
    table(y + 1, x + 7) = _mm256_extract_epi32(vecTable_XSup, 7);

    //FIXME: Passer en vectoriel cette partie
    for (int i = 0; i < VEC_SIZE; i++)
    {
        if (table(y, x + i) >= 4)
            table(y, x + i) = table(y, x) / 4;
    }

    //FIXME: Passer en vectoriel cette partie
    for (int i = 0; i < VEC_SIZE; i++)
    {
        if (table(y, x - i) >= 4)
            table(y, x - i) = table(y, x) / 4;
    }

    // FIXME: voir pour du vectoriel
    if (table(y, x) >= 4)
        table(y, x) %= 4;
    if (table(y, x + 1) >= 4)
        table(y, x + 1) %= 4;
    if (table(y, x + 2) >= 4)
        table(y, x + 2) %= 4;
    if (table(y, x + 3) >= 4)
        table(y, x + 3) %= 4;
    if (table(y, x + 4) >= 4)
        table(y, x + 4) %= 4;
    if (table(y, x + 5) >= 4)
        table(y, x + 5) %= 4;
    if (table(y, x + 6) >= 4)
        table(y, x + 6) %= 4;
    if (table(y, x + 7) >= 4)
        table(y, x + 7) %= 4;

    changement = 1;
}

static void do_tile_vec(int x, int y, int width, int height)
{

    PRINT_DEBUG('c', "tuile [%d-%d][%d-%d] traitée\n", x, x + width - 1, y,
                y + height - 1);

    for (int i = y; i < y + height; i++)
        for (int j = x; j < x + width; j += VEC_SIZE)
        {
            compute_new_state_vec(i, j);
            // compute_multiple_pixels (i, j);
        }
}

#endif