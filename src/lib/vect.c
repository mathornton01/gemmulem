/*
 * Copyright 2022, Micah Thornton and Chanhee Park <parkchanhee@gmail.com>
 *
 * This file is part of GEMMULEM
 *
 * GEMMULEM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * GEMMULEM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GEMMULEM.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <sys/time.h>

#include "vect.h"


void Printm128i(FILE* LogFP, const __m128i v)
{
    int32_t t[4];
    _mm_storeu_si128((__m128i*)t, v);

    fprintf(LogFP, "(%u, %u, %u, %u)\n",
            t[0], t[1], t[2], t[3]);
}

void Printm128d(FILE* LogFP, const __m128d v)
{
    double t[2];
    _mm_storeu_pd(t, v);

    fprintf(LogFP, "(%f, %f)\n",
            t[0], t[1]);
}



int SumVectorI(const int* InVec, size_t Len)
{
    int sum = 0;
    __m256i vsum, v;
    const __m256i vz = _mm256_setzero_si256();
    vsum = _mm256_setzero_si256();

    int i = 0;
    for(; i <= Len - 8; i += 8) {
        v = _mm256_loadu_si256((__m256i*)(InVec + i));
        vsum = _mm256_add_epi32(vsum, v);
    }

    const __m256i vidx = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);

    vsum = _mm256_hadd_epi32(vsum, vz);
    vsum = _mm256_permutevar8x32_epi32(vsum, vidx);
    vsum = _mm256_hadd_epi32(vsum, vz);
    vsum = _mm256_hadd_epi32(vsum, vsum);
    sum = _mm256_cvtsi256_si32(vsum);

    for(; i < Len; i++) {
        sum += InVec[i];
    }
    return sum;
}

double SumVectorD(const double* InVec, size_t Len)
{
    double sum = 0;
    __m256d vsum, v;
    const __m256d vz = _mm256_setzero_pd();
    vsum = _mm256_setzero_pd();

    int i = 0;
    for(; i <= Len - 4; i += 4) {
        v = _mm256_loadu_pd(InVec + i);
        vsum = _mm256_add_pd(vsum, v);
    }


#define vidx (0xd8) // 0b1101 1000

    vsum = _mm256_hadd_pd(vsum, vz);
    vsum = _mm256_permute4x64_pd(vsum, vidx);
    vsum = _mm256_hadd_pd(vsum, vsum);
    sum = _mm256_cvtsd_f64(vsum);

    for(; i < Len; i++) {
        sum += InVec[i];
    }

    return sum;
}

double SumVectorMaskD(const double* InVec, size_t Len, uint64_t* Masks)
{
    __m256d vsum, v;

    int i = 0;

    // load masks to __m256i


    return 0;
}

void DivVectorD(const double* SrcA, const double* SrcB, double* Dst, size_t Len)
{
    __m256d vsrc_a, vsrc_b;
    __m256d vdst;

    int i = 0;
    for(; i <= Len - 4; i += 4) {
        vsrc_a = _mm256_loadu_pd(SrcA + i);
        vsrc_b = _mm256_loadu_pd(SrcB + i);
        vdst = _mm256_div_pd(vsrc_a, vsrc_b);
        _mm256_storeu_pd(Dst + i, vdst);
    }
    for(; i < Len; i++) {
        Dst[i] = SrcA[i] / SrcB[i];
    }
}

void DivVectorValD(const double* SrcA, const double SrcB, double* Dst, size_t Len)
{
#if 0
    __m128d vsrc_a, vsrc_b;
    __m128d vdst;

    vsrc_b = _mm_set_pd(src_b, src_b);

    int i = 0;
    for(; i < src_a.size() - 2; i += 2) {
        vsrc_a = _mm_loadu_pd(data_a + i);
        vdst = _mm_div_pd(vsrc_a, vsrc_b);
        _mm_storeu_pd(data_dst + i, vdst);
    }
#else
    __m256d vsrc_a, vsrc_b;
    __m256d vdst;

    vsrc_b = _mm256_set1_pd(SrcB);

    int i = 0;
    for(; i <= Len - 4; i += 4) {
        vsrc_a = _mm256_loadu_pd(SrcA + i);
        vdst = _mm256_div_pd(vsrc_a, vsrc_b);
        _mm256_storeu_pd(Dst + i, vdst);
    }
#endif
    for(; i < Len; i++) {
        Dst[i] = SrcA[i] / SrcB;
    }
}

void MulVectorD(const double* SrcA, const double* SrcB, double* Dst, size_t Len)
{
    __m256d vsrc_a, vsrc_b;
    __m256d vdst;

    int i = 0;
    for(; i <= Len - 4; i += 4) {
        vsrc_a = _mm256_loadu_pd(SrcA + i);
        vsrc_b = _mm256_loadu_pd(SrcB + i);
        vdst = _mm256_mul_pd(vsrc_a, vsrc_b);
        _mm256_storeu_pd(Dst + i, vdst);
    }
    for(; i < Len; i++) {
        Dst[i] = SrcA[i] * SrcB[i];
    }
}

void MulVectorValD(const double* SrcA, const double SrcB, double* Dst, size_t Len)
{
    __m256d vsrc_a, vsrc_b;
    __m256d vdst;

    vsrc_b = _mm256_set1_pd(SrcB);

    int i = 0;
    for(; i <= Len - 4; i += 4) {
        vsrc_a = _mm256_loadu_pd(SrcA + i);
        vdst = _mm256_mul_pd(vsrc_a, vsrc_b);
        _mm256_storeu_pd(Dst + i, vdst);
    }
    for(; i < Len; i++) {
        Dst[i] = SrcA[i] * SrcB;
    }
}

double GetRelErrVectorD(const double* SrcA, const double* SrcB, size_t Len)
{
    __m256d vsrc_a, vsrc_b;
    __m256d vdst;

    double sum = 0;
    __m256d vsum;
    vsum = _mm256_setzero_pd();

    int i = 0;
    for(; i <= Len - 4; i += 4) {
        __m256d tmp;
        vsrc_a = _mm256_loadu_pd(SrcA + i);
        vsrc_b = _mm256_loadu_pd(SrcB + i);
        tmp = _mm256_sub_pd(vsrc_a, vsrc_b);
        vdst = _mm256_mul_pd(tmp, tmp);
        vsum = _mm256_add_pd(vsum, vdst);
    }

    vsum = _mm256_hadd_pd(vsum, vsum);
#define PERMUTE_IDX (0xd8) // 0b11011000
    vsum = _mm256_permute4x64_pd(vsum, PERMUTE_IDX);
    vsum = _mm256_hadd_pd(vsum, vsum);
    sum = _mm256_cvtsd_f64(vsum);


    for(; i < Len; i++) {
        sum +=  (SrcA[i] - SrcB[i]) * (SrcA[i] - SrcB[i]);
    }
    return sum;
}
