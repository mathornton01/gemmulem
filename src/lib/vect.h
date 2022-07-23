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

#ifndef __VECT_H__
#define __VECT_H__

#include <immintrin.h>

#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus


void Printm128i(FILE* LogFP, const __m128i v);
void Printm128d(FILE* LogFP, const __m128d v);

/**
 *
 * @param InVec
 * @param Len
 */
int SumVectorI(const int* InVec, size_t Len);
double SumVectorD(const double* InVec, size_t Len);


/**
 *
 * @param InVec
 * @param Len
 * @param Masks
 */
double SumVectorMaskD(const double* InVec, size_t Len, uint64_t* Masks);


/**
 * Dst = SrcA / SrcB
 *
 * @param SrcA
 * @param SrcB
 * @param Dst
 * @param Len
 *
 */
void DivVectorValD(const double* SrcA, const double SrcB, double* Dst, size_t Len);
void DivVectorD(const double* SrcA, const double* SrcB, double* Dst, size_t Len);

void MulVectorValD(const double* SrcA, const double SrcB, double* Dst, size_t Len);
void MulVectorD(const double* SrcA, const double* SrcB, double* Dst, size_t Len);

/**
 * returns (SrcA - SrcB)^2
 *
 * @param SrcA
 * @param SrcB
 * @param Len
 *
 * @return
 *
 */
double GetRelErrVectorD(const double* SrcA, const double* SrcB, size_t Len);


#ifdef __cplusplus
}
#endif // __cplusplus


#endif /* __VECT_H__ */
