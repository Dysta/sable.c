#include "easypap.h"

#include <omp.h>
#include <stdbool.h>
#include <assert.h>
#include <immintrin.h>

static uint64_t *TABLE = NULL;
static uint64_t max_grains;

// on ajoute un offset sur le tableau pour gérer les overflow de manière caché
// utilisé par toute les versions. Permet de compute sur les tuiles de taille normal
// tout en ayant des adresses alignés pour la version vectoriel
// on passe donc à un tableau qui ressemble à ceci :
/* Ici DIM = 8
     01234567
 ................
0....xxxxyyyy....
1....xxxxyyyy....
2....xxxxyyyy....
3....xxxxyyyy....
4....xxxxyyyy....
5....xxxxyyyy....
6....xxxxyyyy....
7....xxxxyyyy....
 ................
*/
#define LEFT_OFFSET     4
#define RIGHT_OFFSET    4
#define UP_OFFSET       1
#define DOWN_OFFSET     1

#define table(X, Y) TABLE[((LEFT_OFFSET) + DIM + (RIGHT_OFFSET)) * ((X) + 1) + ((LEFT_OFFSET) + (Y))]
static inline uint64_t* table_addr(const int i, const int j) {
    return &(table(i, j));
}

#define RGB(r, v, b) (((r) << 24 | (v) << 16 | (b) << 8) | 255)

void sable_init()
{
    // on fait un alloc aligné sur 32 bit pour pouvoir utiliser load au lieu de loadu pour haut et bas
    TABLE = aligned_alloc(32, (LEFT_OFFSET + DIM + RIGHT_OFFSET) * (UP_OFFSET + DIM + DOWN_OFFSET) * sizeof(uint64_t));
    memset(TABLE, 0, (LEFT_OFFSET + DIM + RIGHT_OFFSET) * (UP_OFFSET + DIM + DOWN_OFFSET) * sizeof(uint64_t));
}

void sable_finalize()
{
    free(TABLE);
}

///////////////////////////// Production d'une image
void sable_refresh_img()
{
    uint64_t max = 0;
    for (int i = 1; i < DIM - 1; i++)
        for (int j = 1; j < DIM - 1; j++)
        {
            
            int g = table(i, j);

            int r, v, b;
            r = 53;
            v = 43;
            b = 40;
            if (g == 1) {
                r = 129;
                v = 79;
                b = 40;
            }
            else if (g == 2){
                r = 198;
                v = 123;
                b = 75;
            }
            else if (g == 3){
                r = 245;
                v = 205;
                b = 162;
            }
            else if (g == 4) {
                r = v = b = 253;
            }
            else if (g > 4){
                r = 175;
                v = 154;
                b = 148;
            }

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
        const unsigned x = 1 + random() % (DIM - 2);
        const unsigned y = 1 + random() % (DIM - 2);
        const unsigned g = 1000 + (random() % (4000));
        table(x, y) = g;
    }
}

// calcule des grains de sable
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

static unsigned compute_new_state_vec(int y, int x)
{
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

// séparation en tuile
static unsigned do_tile(int x, int y, int width, int height, int who)
{
    PRINT_DEBUG('c', "tuile [%d-%d][%d-%d] traitée\n", x, x + width - 1, y, y + height - 1);

    monitoring_start_tile(who);
    unsigned changes = 0;
    for (int i = y; i < y + height; i++)
        for (int j = x; j < x + width; j++) {
            changes += compute_new_state(i, j);
        }
    monitoring_end_tile(x, y, width, height, who);

    return changes;
}

static unsigned do_tile2(int x, int y, int width, int height, int who)
{
    PRINT_DEBUG('c', "tuile [%d-%d][%d-%d] traitée\n", x, x + width - 1, y, y + height - 1);

    unsigned changes = 0;
    monitoring_start_tile(who);
    for (int i = y; i < y + height; i++) {
        for (int j = x; j < x + width; j++) {
            changes += compute_new_state(i, j);
        }
    }

    // si aucun changement, pas besoin d'un second balayage
    if (changes == 0) return 0;

    for (int i = y; i < y + height; i++) {
        for (int j = x; j < x + width; j++) {
            changes += compute_new_state(i, j);
        }
    }
    monitoring_end_tile(x, y, width, height, who);

    return changes;
}

static unsigned do_tile_vec(int x, int y, int width, int height, int who)
{

    PRINT_DEBUG('c', "tuile [%d-%d][%d-%d] traitée\n", x, x + width - 1, y,
                y + height - 1);

    monitoring_start_tile(who);
    unsigned changes = 0;
    for (int i = y; i < y + height; i++) {
        for (int j = x; j < x + width; j += VEC_SIZE/2) {
            changes += compute_new_state_vec(i, j);
        }
    }
    monitoring_end_tile(x, y, width, height, who);

    return changes;
}

static unsigned do_tile_vec2(int x, int y, int width, int height, int who)
{

    PRINT_DEBUG('c', "tuile [%d-%d][%d-%d] traitée\n", x, x + width - 1, y,
                y + height - 1);

    unsigned changes = 0, diff;
    monitoring_start_tile(who);
    for (int i = y; i < y + height; i++) {
        for (int j = x; j < x + width; j += VEC_SIZE/2) {
            diff = compute_new_state_vec(i, j);
            changes += diff;

            if (diff == 0) continue;

            changes += compute_new_state_vec(i, j);
        }
    }
    monitoring_end_tile(x, y, width, height, who);

    return changes;
}

// différentes version du noyau

///////////////////////////// Version séquentielle (seq)
// ./run -k sable -v vec
unsigned sable_compute_seq(unsigned nb_iter)
{
    unsigned changes = 0;
    for (unsigned it = 1; it <= nb_iter; it++)
    {
        // On traite toute l'image en un coup (oui, c'est une grosse tuile)
        changes += do_tile(0, 0, DIM, DIM, 0);
        if (changes == 0)
            return it;
        changes = 0;
    }
    return 0;
}

///////////////////////////// Version parallèle tuilée en bande (ompfor)
// ./run -k sable -v ompfor
unsigned sable_compute_ompfor(unsigned nb_iter)
{
    unsigned changes = 0;
    const int max_tile_idx = (DIM / TILE_SIZE + (DIM % TILE_SIZE > 0));

    for (unsigned it = 1; it <= nb_iter; it++)
    {
        #pragma omp parallel for reduction(+:changes) schedule(runtime)
        for (int y = 0; y < max_tile_idx; y+=2)
        {
            const int x_start   = 0;
            const int y_start   = y * TILE_SIZE;
            const int width     = DIM;
            const int height    = (y == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
            
            changes += do_tile(x_start, y_start, width, height, omp_get_thread_num());
        }

        #pragma omp parallel for reduction(+:changes) schedule(runtime)
        for (int y = 1; y < max_tile_idx; y+=2)
        {
            const int x_start   = 0;
            const int y_start   = y * TILE_SIZE;
            const int width     = DIM;
            const int height    = (y == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
            
            changes += do_tile(x_start, y_start, width, height, omp_get_thread_num());
        }

        if (changes == 0)
            return it;

        changes = 0;
    }

    return 0;
}

///////////////////////////// Version parallèle tuilée avec double balayage (ompfor2)
// ./run -k sable -v ompfor2
unsigned sable_compute_ompfor2(unsigned nb_iter)
{
    unsigned changes = 0;
    const int max_tile_idx = (DIM / TILE_SIZE + (DIM % TILE_SIZE > 0));

    for (unsigned it = 1; it <= nb_iter; it++)
    {
        #pragma omp parallel for reduction(+:changes) schedule(runtime)
        for (int y = 0; y < max_tile_idx; y+=2)
        {
            const int x_start   = 0;
            const int y_start   = y * TILE_SIZE;
            const int width     = DIM;
            const int height    = (y == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
            
            changes += do_tile2(x_start, y_start, width, height, omp_get_thread_num());
        }

        #pragma omp parallel for reduction(+:changes) schedule(runtime)
        for (int y = 1; y < max_tile_idx; y+=2)
        {
            const int x_start   = 0;
            const int y_start   = y * TILE_SIZE;
            const int width     = DIM;
            const int height    = (y == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
            
            changes += do_tile2(x_start, y_start, width, height, omp_get_thread_num());
        }

        if (changes == 0)
            return it;

        changes = 0;
    }

    return 0;
}

///////////////////////////// Version parallèle tuilée en damié (ompfor_tiled)
// ./run -k sable -v ompfor_tiled
unsigned sable_compute_ompfor_tiled(unsigned nb_iter)
{
    const int max_tile_idx = (DIM / TILE_SIZE + (DIM % TILE_SIZE > 0));
    unsigned changes = 0;

    for (unsigned it = 1; it <= nb_iter; it++) {
        #pragma omp parallel for reduction(+:changes) schedule(runtime) collapse(2)
        for (int y = 0; y < max_tile_idx; y+=2){
            for (int x = 0; x < max_tile_idx; x +=2){
                const int y_start   = y * TILE_SIZE;
                const int x_start   = x * TILE_SIZE;
                const int width     = (x == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
                const int height    = (y == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
                
                changes += do_tile(x_start, y_start, width, height, omp_get_thread_num());
            }
        }
        
        #pragma omp parallel for reduction(+:changes) schedule(runtime) collapse(2)
        for (int y = 1; y < max_tile_idx; y+=2){
            for (int x = 1; x < max_tile_idx; x += 2){
                const int y_start   = y * TILE_SIZE;
                const int x_start   = x * TILE_SIZE;
                const int width     = (x == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
                const int height    = (y == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
                
                changes += do_tile(x_start, y_start, width, height, omp_get_thread_num());
            }
        }
        #pragma omp parallel for reduction(+:changes) schedule(runtime) collapse(2)
        for (int y = 0; y < max_tile_idx; y+=2){
            for (int x = 1; x < max_tile_idx; x +=2){
                const int y_start   = y * TILE_SIZE;
                const int x_start   = x * TILE_SIZE;
                const int width     = (x == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
                const int height    = (y == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
                
                changes += do_tile(x_start, y_start, width, height, omp_get_thread_num());
            }
        }
        
        #pragma omp parallel for reduction(+:changes) schedule(runtime) collapse(2)
        for (int y = 1; y < max_tile_idx; y+=2){
            for (int x = 0; x < max_tile_idx; x += 2){
                const int y_start   = y * TILE_SIZE;
                const int x_start   = x * TILE_SIZE;
                const int width     = (x == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
                const int height    = (y == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
                
                changes += do_tile(x_start, y_start, width, height, omp_get_thread_num());
            }
        }
            
        if (changes == 0)
            return it;
        
        changes = 0;
    }

    return 0;
}

///////////////////////////// Version parallèle tuilée en damié avec double balayage (ompfor_tiled2)
// ./run -k sable -v ompfor_tiled2
unsigned sable_compute_ompfor_tiled2(unsigned nb_iter)
{
    const int max_tile_idx = (DIM / TILE_SIZE + (DIM % TILE_SIZE > 0));
    unsigned changes = 0;

    for (unsigned it = 1; it <= nb_iter; it++) {
        #pragma omp parallel for reduction(+:changes) schedule(runtime) collapse(2)
        for (int y = 0; y < max_tile_idx; y+=2){
            for (int x = 0; x < max_tile_idx; x +=2){
                const int y_start   = y * TILE_SIZE;
                const int x_start   = x * TILE_SIZE;
                const int width     = (x == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
                const int height    = (y == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
                
                changes += do_tile2(x_start, y_start, width, height, omp_get_thread_num());
            }
        }
        
        #pragma omp parallel for reduction(+:changes) schedule(runtime) collapse(2)
        for (int y = 1; y < max_tile_idx; y+=2){
            for (int x = 1; x < max_tile_idx; x += 2){
                const int y_start   = y * TILE_SIZE;
                const int x_start   = x * TILE_SIZE;
                const int width     = (x == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
                const int height    = (y == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
                
                changes += do_tile2(x_start, y_start, width, height, omp_get_thread_num());
            }
        }
        #pragma omp parallel for reduction(+:changes) schedule(runtime) collapse(2)
        for (int y = 0; y < max_tile_idx; y+=2){
            for (int x = 1; x < max_tile_idx; x +=2){
                const int y_start   = y * TILE_SIZE;
                const int x_start   = x * TILE_SIZE;
                const int width     = (x == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
                const int height    = (y == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
                
                changes += do_tile2(x_start, y_start, width, height, omp_get_thread_num());
            }
        }
        
        #pragma omp parallel for reduction(+:changes) schedule(runtime) collapse(2)
        for (int y = 1; y < max_tile_idx; y+=2){
            for (int x = 0; x < max_tile_idx; x += 2){
                const int y_start   = y * TILE_SIZE;
                const int x_start   = x * TILE_SIZE;
                const int width     = (x == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
                const int height    = (y == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
                
                changes += do_tile2(x_start, y_start, width, height, omp_get_thread_num());
            }
        }
            
        if (changes == 0)
            return it;
        
        changes = 0;
    }

    return 0;
}

// compute vectoriel

///////////////////////////// Version vectorielle sequentielle (vec)
// ./run -k sable -v vec
unsigned sable_compute_vec(unsigned nb_iter)
{
    if (DIM % 8 != 0) {
        fprintf(stderr, "Error: DIM (%d) must be a multiple of 8 for vectorial compute\n", DIM);
        assert(DIM % 8 == 0);
    }

    unsigned changes = 0;
    for (unsigned it = 1; it <= nb_iter; it++)
    {
        // On traite toute l'image en un coup (oui, c'est une grosse tuile)
        changes += do_tile_vec(0, 0, DIM, DIM, 0);
        if (changes == 0)
            return it;
        changes = 0;
    }
    return 0;
}

///////////////////////////// Version vectorielle sequentielle avec double balayage (vec2)
// ./run -k sable -v vec2
unsigned sable_compute_vec2(unsigned nb_iter)
{
    if (DIM % 8 != 0) {
        fprintf(stderr, "Error: DIM (%d) must be a multiple of 8 for vectorial compute\n", DIM);
        assert(DIM % 8 == 0);
    }

    unsigned changes = 0;
    for (unsigned it = 1; it <= nb_iter; it++)
    {
        // On traite toute l'image en un coup (oui, c'est une grosse tuile)
        changes += do_tile_vec2(0, 0, DIM, DIM, 0);
        if (changes == 0)
            return it;
        changes = 0;
    }
    return 0;
}

///////////////////////////// Version vectorielle parallèle (vec_ompfor)
// ./run -k sable -v vec_ompfor
unsigned sable_compute_vec_ompfor(unsigned nb_iter)
{
    if (DIM % 8 != 0) {
        fprintf(stderr, "Error: DIM (%d) must be a multiple of 8 for vectorial compute\n", DIM);
        assert(DIM % 8 == 0);
    }

    unsigned changes = 0;
    const int max_tile_idx = (DIM / TILE_SIZE + (DIM % TILE_SIZE > 0));

    for (unsigned it = 1; it <= nb_iter; it++)
    {
        #pragma omp parallel for reduction(+:changes) schedule(runtime)
        for (int y = 0; y < max_tile_idx; y+=2)
        {
            const int x_start   = 0;
            const int y_start   = y * TILE_SIZE;
            const int width     = DIM;
            const int height    = (y == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
            
            changes += do_tile_vec(x_start, y_start, width, height, omp_get_thread_num());
        }

        #pragma omp parallel for reduction(+:changes) schedule(runtime)
        for (int y = 1; y < max_tile_idx; y+=2)
        {
            const int x_start   = 0;
            const int y_start   = y * TILE_SIZE;
            const int width     = DIM;
            const int height    = (y == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
            
            changes += do_tile_vec(x_start, y_start, width, height, omp_get_thread_num());
        }

        if (changes == 0)
            return it;

        changes = 0;
    }
    return 0;
}

///////////////////////////// Version vectorielle parallèle avec double balayage (vec_ompfor2)
// ./run -k sable -v vec_ompfor2
unsigned sable_compute_vec_ompfor2(unsigned nb_iter)
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
            const int x_start   = 0;
            const int y_start   = y * TILE_SIZE;
            const int width     = DIM;
            const int height    = (y == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
            
            changes += do_tile_vec2(x_start, y_start, width, height, omp_get_thread_num());
        }

        #pragma omp parallel for reduction(+:changes) schedule(runtime)
        for (int y = 1; y < max_tile_idx; y+=2)
        {
            const int x_start   = 0;
            const int y_start   = y * TILE_SIZE;
            const int width     = DIM;
            const int height    = (y == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
            
            changes += do_tile_vec2(x_start, y_start, width, height, omp_get_thread_num());
        }

        if (changes == 0)
            return it;

        changes = 0;
    }
    return 0;
}

///////////////////////////// Version vectoriel tuilée en damié avec double balayage (vec_ompfor_tiled)
// ./run -k sable -v vec_ompfor_tiled
unsigned sable_compute_vec_ompfor_tiled(unsigned nb_iter)
{
    const int max_tile_idx = (DIM / TILE_SIZE + (DIM % TILE_SIZE > 0));
    unsigned changes = 0;

    for (unsigned it = 1; it <= nb_iter; it++) {
        #pragma omp parallel for reduction(+:changes) schedule(runtime) collapse(2)
        for (int y = 0; y < max_tile_idx; y+=2){
            for (int x = 0; x < max_tile_idx; x +=2){
                const int y_start   = y * TILE_SIZE;
                const int x_start   = x * TILE_SIZE;
                const int width     = (x == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
                const int height    = (y == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
                
                changes += do_tile_vec(x_start, y_start, width, height, omp_get_thread_num());
            }
        }
        
        #pragma omp parallel for reduction(+:changes) schedule(runtime) collapse(2)
        for (int y = 1; y < max_tile_idx; y+=2){
            for (int x = 1; x < max_tile_idx; x += 2){
                const int y_start   = y * TILE_SIZE;
                const int x_start   = x * TILE_SIZE;
                const int width     = (x == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
                const int height    = (y == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
                
                changes += do_tile_vec(x_start, y_start, width, height, omp_get_thread_num());
            }
        }
        #pragma omp parallel for reduction(+:changes) schedule(runtime) collapse(2)
        for (int y = 0; y < max_tile_idx; y+=2){
            for (int x = 1; x < max_tile_idx; x +=2){
                const int y_start   = y * TILE_SIZE;
                const int x_start   = x * TILE_SIZE;
                const int width     = (x == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
                const int height    = (y == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
                
                changes += do_tile_vec(x_start, y_start, width, height, omp_get_thread_num());
            }
        }
        
        #pragma omp parallel for reduction(+:changes) schedule(runtime) collapse(2)
        for (int y = 1; y < max_tile_idx; y+=2){
            for (int x = 0; x < max_tile_idx; x += 2){
                const int y_start   = y * TILE_SIZE;
                const int x_start   = x * TILE_SIZE;
                const int width     = (x == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
                const int height    = (y == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
                
                changes += do_tile_vec(x_start, y_start, width, height, omp_get_thread_num());
            }
        }
            
        if (changes == 0)
            return it;
        
        changes = 0;
    }

    return 0;
}

///////////////////////////// Version vectoriel tuilée en damié avec double balayage (vec_ompfor_tiled2)
// ./run -k sable -v vec_ompfor_tiled2
unsigned sable_compute_vec_ompfor_tiled2(unsigned nb_iter)
{
    const int max_tile_idx = (DIM / TILE_SIZE + (DIM % TILE_SIZE > 0));
    unsigned changes = 0;

    for (unsigned it = 1; it <= nb_iter; it++) {
        #pragma omp parallel for reduction(+:changes) schedule(runtime) collapse(2)
        for (int y = 0; y < max_tile_idx; y+=2){
            for (int x = 0; x < max_tile_idx; x +=2){
                const int y_start   = y * TILE_SIZE;
                const int x_start   = x * TILE_SIZE;
                const int width     = (x == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
                const int height    = (y == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
                
                changes += do_tile_vec2(x_start, y_start, width, height, omp_get_thread_num());
            }
        }
        
        #pragma omp parallel for reduction(+:changes) schedule(runtime) collapse(2)
        for (int y = 1; y < max_tile_idx; y+=2){
            for (int x = 1; x < max_tile_idx; x += 2){
                const int y_start   = y * TILE_SIZE;
                const int x_start   = x * TILE_SIZE;
                const int width     = (x == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
                const int height    = (y == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
                
                changes += do_tile_vec2(x_start, y_start, width, height, omp_get_thread_num());
            }
        }
        #pragma omp parallel for reduction(+:changes) schedule(runtime) collapse(2)
        for (int y = 0; y < max_tile_idx; y+=2){
            for (int x = 1; x < max_tile_idx; x +=2){
                const int y_start   = y * TILE_SIZE;
                const int x_start   = x * TILE_SIZE;
                const int width     = (x == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
                const int height    = (y == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
                
                changes += do_tile_vec2(x_start, y_start, width, height, omp_get_thread_num());
            }
        }
        
        #pragma omp parallel for reduction(+:changes) schedule(runtime) collapse(2)
        for (int y = 1; y < max_tile_idx; y+=2){
            for (int x = 0; x < max_tile_idx; x += 2){
                const int y_start   = y * TILE_SIZE;
                const int x_start   = x * TILE_SIZE;
                const int width     = (x == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
                const int height    = (y == max_tile_idx - 1 && DIM % TILE_SIZE != 0) ? DIM % TILE_SIZE : TILE_SIZE;
                
                changes += do_tile_vec2(x_start, y_start, width, height, omp_get_thread_num());
            }
        }
            
        if (changes == 0)
            return it;
        
        changes = 0;
    }

    return 0;
}