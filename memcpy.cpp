#include <iostream>
#include <emmintrin.h>
#include <cstring>
#include <immintrin.h>

constexpr int num = 50000;
constexpr size_t size = 6000;

extern "C"
char *get_stringish_seq(size_t *soft_limit) {
    size_t sz = rand() % (*soft_limit) + 10;
    *soft_limit = sz + 1;
    char *s = (char *) malloc(sz + 1);
    for (size_t i = 0; i < sz; i++) {
        s[i] = (char) (rand() % 70 + 31); // ascii humanreadable strings
    }
    s[sz] = '\0';

    return s;
}

extern "C"
void memcpy_sf(void *dst, void const *src, size_t sz) {
    char *d = (char *) dst;
    char *s = (char *) src;
    while (sz--) {
        *d++ = *s++;
    }
}

extern "C"
void memcpy_vec(void *dst, void const *src, size_t sz) {
    if (sz <= 64) {
        return memcpy_sf(dst, src, sz);
    }

    size_t startp = 0;
    char *d = (char *) dst;
    char *s = (char *) src;
    while ((((size_t) dst + startp) & 31) != 0) {
        *d++ = *s++;
        startp++;
    }

    __m256i out_temp;
    size_t endp = (sz - startp) & 31;
    for (size_t i = startp; i < (sz - endp); i += 32) {
        __asm__ volatile (
                "vmovdqu        %0, [%1]\t\n"
                "vmovntdq       [%2], %0\t\n"
                "add            %2, 32\t\n"
                "add            %3, 32\t\n"
        :"=v"(out_temp), "+r"(s), "+r"(d)
        :
        :"memory");
    }

    for (size_t i = sz - endp; i < sz; i++) {
        *d++ = *s++;
    }
    _mm_sfence(); // nontemporal commit
}

extern "C"
int main() {
    srand(time(NULL));

    clock_t total_sf = 0, total_vec = 0;

    for (int i = 0; i < num; i++) {
        size_t asize = size;

        char *src = get_stringish_seq(&asize);
        char *dst_s = (char *) malloc(asize);
        char *dst_v = (char *) malloc(asize);

        clock_t s;
        clock_t f;

        s = clock();
        memcpy_sf(dst_s, src, asize);
        f = clock();

        total_sf += (f - s);

        s = clock();
        memcpy_vec(dst_v, src, asize);
        f = clock();

        total_vec += (f - s);

        if (memcmp(dst_s, src, asize) | memcmp(dst_v, src, asize)) {
            printf("!mismatch\n%s\n%s\n%s\n\n", src, dst_s, dst_v);
        }

        free(src);
        free(dst_s);
        free(dst_v);
    }

    printf("done\nstraight: %lld\nfast: %lld\nspeedz: %lf\n",
           (long long int) total_sf,
           (long long int) total_vec,
           ((double) total_sf / (double) total_vec));
}