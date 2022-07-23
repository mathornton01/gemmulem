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

#include "simde/x86/avx2.h"

#include "vect.h"

void Printm128i(FILE* LogFP, const simde__m128i v);
void Printm128d(FILE* LogFP, const simde__m128d v);


void Printm128i(FILE* LogFP, const simde__m128i v)
{
    int32_t t[4];
    simde_mm_storeu_si128((simde__m128i*)t, v);

    fprintf(LogFP, "(%u, %u, %u, %u)\n",
            t[0], t[1], t[2], t[3]);
}

void Printm128d(FILE* LogFP, const simde__m128d v)
{
    double t[2];
    simde_mm_storeu_pd(t, v);

    fprintf(LogFP, "(%f, %f)\n",
            t[0], t[1]);
}



int SumVectorI(const int* InVec, size_t Len)
{
    int sum = 0;
    simde__m256i vsum, v;
    const simde__m256i vz = simde_mm256_setzero_si256();
    vsum = simde_mm256_setzero_si256();

    int i = 0;
    for(; i <= Len - 8; i += 8) {
        v = simde_mm256_loadu_si256((simde__m256i*)(InVec + i));
        vsum = simde_mm256_add_epi32(vsum, v);
    }

    const simde__m256i vidx = simde_mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);

    vsum = simde_mm256_hadd_epi32(vsum, vz);
    vsum = simde_mm256_permutevar8x32_epi32(vsum, vidx);
    vsum = simde_mm256_hadd_epi32(vsum, vz);
    vsum = simde_mm256_hadd_epi32(vsum, vsum);
    sum = simde_mm256_cvtsi256_si32(vsum);

    for(; i < Len; i++) {
        sum += InVec[i];
    }
    return sum;
}

double SumVectorD(const double* InVec, size_t Len)
{
    double sum = 0;
    simde__m256d vsum, v;
    const simde__m256d vz = simde_mm256_setzero_pd();
    vsum = simde_mm256_setzero_pd();

    int i = 0;
    for(; i <= Len - 4; i += 4) {
        v = simde_mm256_loadu_pd(InVec + i);
        vsum = simde_mm256_add_pd(vsum, v);
    }


#define vidx (0xd8) // 0b1101 1000

    vsum = simde_mm256_hadd_pd(vsum, vz);
    vsum = simde_mm256_permute4x64_pd(vsum, vidx);
    vsum = simde_mm256_hadd_pd(vsum, vsum);
    sum = simde_mm256_cvtsd_f64(vsum);

    for(; i < Len; i++) {
        sum += InVec[i];
    }

    return sum;
}

double SumVectorMaskD(const double* InVec, size_t Len, uint64_t* Masks)
{
    simde__m256d vsum, v;

    int i = 0;

    // load masks to __m256i

    return 0;
}

void DivVectorD(const double* SrcA, const double* SrcB, double* Dst, size_t Len)
{
    simde__m256d vsrc_a, vsrc_b;
    simde__m256d vdst;

    int i = 0;
    for(; i <= Len - 4; i += 4) {
        vsrc_a = simde_mm256_loadu_pd(SrcA + i);
        vsrc_b = simde_mm256_loadu_pd(SrcB + i);
        vdst = simde_mm256_div_pd(vsrc_a, vsrc_b);
        simde_mm256_storeu_pd(Dst + i, vdst);
    }
    for(; i < Len; i++) {
        Dst[i] = SrcA[i] / SrcB[i];
    }
}

void DivVectorValD(const double* SrcA, const double SrcB, double* Dst, size_t Len)
{
    simde__m256d vsrc_a, vsrc_b;
    simde__m256d vdst;

    vsrc_b = simde_mm256_set1_pd(SrcB);

    int i = 0;
    for(; i <= Len - 4; i += 4) {
        vsrc_a = simde_mm256_loadu_pd(SrcA + i);
        vdst = simde_mm256_div_pd(vsrc_a, vsrc_b);
        simde_mm256_storeu_pd(Dst + i, vdst);
    }
    for(; i < Len; i++) {
        Dst[i] = SrcA[i] / SrcB;
    }
}

void MulVectorD(const double* SrcA, const double* SrcB, double* Dst, size_t Len)
{
    simde__m256d vsrc_a, vsrc_b;
    simde__m256d vdst;

    int i = 0;
    for(; i <= Len - 4; i += 4) {
        vsrc_a = simde_mm256_loadu_pd(SrcA + i);
        vsrc_b = simde_mm256_loadu_pd(SrcB + i);
        vdst = simde_mm256_mul_pd(vsrc_a, vsrc_b);
        simde_mm256_storeu_pd(Dst + i, vdst);
    }
    for(; i < Len; i++) {
        Dst[i] = SrcA[i] * SrcB[i];
    }
}

void MulVectorValD(const double* SrcA, const double SrcB, double* Dst, size_t Len)
{
    simde__m256d vsrc_a, vsrc_b;
    simde__m256d vdst;

    vsrc_b = simde_mm256_set1_pd(SrcB);

    int i = 0;
    for(; i <= Len - 4; i += 4) {
        vsrc_a = simde_mm256_loadu_pd(SrcA + i);
        vdst = simde_mm256_mul_pd(vsrc_a, vsrc_b);
        simde_mm256_storeu_pd(Dst + i, vdst);
    }
    for(; i < Len; i++) {
        Dst[i] = SrcA[i] * SrcB;
    }
}

double GetRelErrVectorD(const double* SrcA, const double* SrcB, size_t Len)
{
    simde__m256d vsrc_a, vsrc_b;
    simde__m256d vdst;

    double sum = 0;
    simde__m256d vsum;
    vsum = simde_mm256_setzero_pd();

    int i = 0;
    for(; i <= Len - 4; i += 4) {
        simde__m256d tmp;
        vsrc_a = simde_mm256_loadu_pd(SrcA + i);
        vsrc_b = simde_mm256_loadu_pd(SrcB + i);
        tmp = simde_mm256_sub_pd(vsrc_a, vsrc_b);
        vdst = simde_mm256_mul_pd(tmp, tmp);
        vsum = simde_mm256_add_pd(vsum, vdst);
    }

    vsum = simde_mm256_hadd_pd(vsum, vsum);
#define PERMUTE_IDX (0xd8) // 0b11011000
    vsum = simde_mm256_permute4x64_pd(vsum, PERMUTE_IDX);
    vsum = simde_mm256_hadd_pd(vsum, vsum);
    sum = simde_mm256_cvtsd_f64(vsum);


    for(; i < Len; i++) {
        sum +=  (SrcA[i] - SrcB[i]) * (SrcA[i] - SrcB[i]);
    }
    return sum;
}
