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

#ifdef _MSC_VER
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#else
#include <sys/time.h>
#endif

#include "EM.h"
#include "vect.h"

// Some simple utility functions I need.
double Min(const double* ValuePtr, size_t Size);
double Max(const double* ValuePtr, size_t Size);
double Mean(const double* ValuePtr, size_t Size);
double Var(const double* ValuePtr, size_t Size);
double GetNormLH(double value, double mean, double var);
double GetExpoLH(double value, double mn);

int RandomInitGaussianEM(const double* ValuePtr, size_t Size, int NumGaussians, EMResultGaussian_t* ResultPtr);
int RandomInitExponentialEM(const double* ValuePtr, size_t Size, int NumExponentials, EMResultExponential_t* ResultPtr);

// Default EMConfig_t
EMConfig_t EMConfigDefault = {
        .verbose = 0,
        .maxiter = 1000,
        .rtole = 0.00001,
};
EMConfig_t EMConfigGaussianDefault = {
        .verbose = 0,
        .maxiter = 1000,
        .rtole = 0.000001,
};
EMConfig_t EMConfigExponentialDefault = {
        .verbose = 0,
        .maxiter = 1000,
        .rtole = 0.000001,
};


/**
 * Get current timestamp in microsecond
 * @return
 */
static unsigned long long GetCurrentTimestamp()
{
#ifdef _MSC_VER
    // Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
    // This magic number is the number of 100 nanosecond intervals since January 1, 1601 (UTC)
    // until 00:00:00 January 1, 1970 
    static const uint64_t EPOCH = ((uint64_t)116444736000000000ULL);

    SYSTEMTIME  system_time;
    FILETIME    file_time;
    uint64_t    time;

    GetSystemTime(&system_time);
    SystemTimeToFileTime(&system_time, &file_time);
    time = ((uint64_t)file_time.dwLowDateTime);
    time += ((uint64_t)file_time.dwHighDateTime) << 32;

    return time;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000000ULL + tv.tv_usec;
#endif
}

int UnmixGaussians(const double* ValuePtr, size_t Size, int NumGaussians, EMResultGaussian_t* ResultPtr, EMConfig_t* ConfigPtr)
{
    if (ValuePtr == NULL || ResultPtr == NULL || Size == 0) {
        return -1;
    }

    EMConfig_t cfg = EMConfigGaussianDefault;
    if (ConfigPtr) {
        cfg = *ConfigPtr;
    }

    RandomInitGaussianEM(ValuePtr, Size, NumGaussians, ResultPtr);

    int mconv = 0;
    int vconv = 0;
    int pconv = 0;

    double* mcur = ResultPtr->means_init;
    double* mprev = ResultPtr->means_final;
    double* vcur = ResultPtr->vars_init;
    double* vprev = ResultPtr->vars_final;
    double* pcur = ResultPtr->probs_init;
    double* pprev = ResultPtr->probs_final;

    double merr = 0.0;
    double verr = 0.0;
    double perr = 0.0;
    int iter = 1;

    if (cfg.verbose){
        printf("INFO: Gaussian Mixture Deconvolution by Expectation Maximization\n");
        printf("INFO:  Number of Gaussians involved in mixture - %d\n", NumGaussians);
        printf("INFO:  Number of Values observed total - %lu\n", Size);
    }

    // Likelihood matrix,
    //  Row is NumDist, Col is Size
    double* lhall = NULL;
    lhall = (double *)malloc(sizeof(double) * Size * NumGaussians);
    memset(lhall, 0, sizeof(double) * Size * NumGaussians);

    double* tmpvec = NULL;
    tmpvec = (double *)malloc(sizeof(double) * Size);
    memset(tmpvec, 0, sizeof(double) * Size);

    while ((!mconv || !vconv || !pconv) && iter <= cfg.maxiter) {
        if (cfg.verbose) {
            printf("INFO:  EM Iteration - %d | Rel. Error Mean : %e | Rel. Error Var : %e | Rel. Error Prop : %e\n", iter, merr, verr, perr);
        }

        for (int j = 0; j < NumGaussians; j++){
            const double m = mprev[j];
            const double v = vprev[j];
            const double p = pprev[j];

            for (int i = 0; i < Size; i++){
                double lh = GetNormLH(ValuePtr[i], m, v) * p;

                if (lh == 0) {
                    lh = 0.000001;
                }
                lhall[j * Size + i] = lh;
            }
        }

        for (int i = 0; i < Size; i++) {
            double total = 0;
            for (int j = 0; j < NumGaussians; j++) {
                total += lhall[j * Size + i];
            }
            for (int j = 0; j < NumGaussians; j++) {
                const size_t offset = j * Size + i;
                lhall[offset] = lhall[offset] / total;
            }
        }


        // Generate the new proportion estimates
        for(int j = 0; j < NumGaussians; j++) {
            pcur[j] = SumVectorD(&lhall[j * Size], Size);
            pcur[j] /= Size;
        }

        // Get the error estimate
        perr = 0;
        for (int j = 0; j < NumGaussians; j++){
            perr += sqrt(pow(pcur[j]-pprev[j],2));
        }
        perr /= NumGaussians;

        pconv = (perr < cfg.rtole);

        // Generate the mean estimates
        double colsum;
        for (int j = 0; j < NumGaussians; j++){
            MulVectorD(&lhall[j * Size], ValuePtr, tmpvec, Size);
            mcur[j] = SumVectorD(tmpvec, Size);
            colsum = SumVectorD(&lhall[j * Size], Size);
            mcur[j] /= colsum;
        }

        // Get the error estimate
        merr = 0;
        for (int j = 0; j < NumGaussians; j++){
            merr += sqrt(pow(mcur[j]-mprev[j],2));
        }
        merr /= NumGaussians;

        mconv = (merr < cfg.rtole);


        // Generate the Variance estimates
        for (int j = 0; j < NumGaussians; j++){
            vcur[j] = 0;
            colsum = 0;
            for (int i = 0; i < Size; i++){
                vcur[j] += lhall[j * Size + i] * (pow(ValuePtr[i] - mcur[j], 2));
                colsum += lhall[j * Size + i];
            }
            vcur[j] /= colsum;
        }

        verr = 0;
        for (int j = 0; j < NumGaussians; j++){
            verr += sqrt(pow(vcur[j]-vprev[j],2));
        }
        verr /= NumGaussians;

        vconv = (verr < cfg.rtole);

        // switch cur and prev
        double *tmp = pprev;
        pprev = pcur;
        pcur = tmp;

        tmp = mprev;
        mprev = mcur;
        mcur = tmp;

        tmp = vprev;
        vprev = vcur;
        vcur = tmp;

        iter++;
    }

    ResultPtr->iterstaken = iter;

    ResultPtr->means_init = mcur;
    ResultPtr->means_final = mprev;

    ResultPtr->probs_init = pcur;
    ResultPtr->probs_final = pprev;

    ResultPtr->vars_init = vcur;
    ResultPtr->vars_final = vprev;

    free(lhall);
    free(tmpvec);

    return 0;
}

int RandomInitGaussianEM(const double* ValuePtr, size_t Size, int NumGaussians, EMResultGaussian_t* ResultPtr)
{

    ResultPtr->numGaussians = NumGaussians;
    double minv = Min(ValuePtr, Size);
    double maxv = Max(ValuePtr, Size);
    double meanv = Mean(ValuePtr, Size);
    double varv = Var(ValuePtr, Size);

    size_t VecSize = sizeof(double) * NumGaussians;

    ResultPtr->means_init = (double *)malloc(VecSize);
    ResultPtr->vars_init = (double *)malloc(VecSize);
    ResultPtr->probs_init = (double *)malloc(VecSize);

    ResultPtr->means_final = (double *)malloc(VecSize);
    ResultPtr->vars_final = (double *)malloc(VecSize);
    ResultPtr->probs_final = (double *)malloc(VecSize);

    for (int i = 0; i < NumGaussians; i++) {
        int ri = rand() % 10000;
        double rd = (double)ri / 10000.0;
        double rnmn = (maxv - minv) * rd + minv;
        ResultPtr->means_init[i] = rnmn;

        ri = rand() % 10000;
        rd = (double)ri / 10000.0;
        double rnvr = (varv * 0.8 - varv * 0.4) * rd + varv * 0.4;
        ResultPtr->vars_init[i] = rnvr;

        ResultPtr->probs_init[i] = 1.0 / NumGaussians;
    }

    memcpy(ResultPtr->means_final, ResultPtr->means_init, VecSize);
    memcpy(ResultPtr->vars_final, ResultPtr->vars_init, VecSize);
    memcpy(ResultPtr->probs_final, ResultPtr->probs_init, VecSize);

    return 0;
}

double Min(const double* ValuePtr, size_t Size)
{
    double min = ValuePtr[0];

    for (int i = 1; i < Size; i++) {
        if (min > ValuePtr[i]) {
            min = ValuePtr[i];
        }
    }
    return min;
}

double Max(const double* ValuePtr, size_t Size)
{
    double max = ValuePtr[0];
    for (int i = 1; i < Size; i++) {
        if (max < ValuePtr[i]) {
            max = ValuePtr[i];
        }
    }
    return max;
}

double Mean(const double* ValuePtr, size_t Size)
{
    double mean = ValuePtr[0];

    for(int i = 1; i < Size; i++) {
        mean += ValuePtr[i];
    }
    mean = mean / Size;
    return mean;
}

double Var(const double* ValuePtr, size_t Size)
{
    double mn = Mean(ValuePtr, Size);
    double vr = (ValuePtr[0] - mn) * (ValuePtr[0] - mn);
    for (int i = 1; i < Size; i++) {
        vr = vr + (ValuePtr[i] - mn) * (ValuePtr[i] - mn);
    }
    vr = vr / Size;

    return vr;
}

double GetNormLH(double value, double mean, double var){
    double sd = sqrt(var);
    double LH = (1/(sd*sqrt(2*3.14159265)))*pow(2.71828182,(-0.5)*pow((value-mean)/sd,2));
    return(LH);
}


int UnmixExponentials(const double* ValuePtr, size_t Size, int NumExponentials, EMResultExponential_t* ResultPtr, EMConfig_t* ConfigPtr)
{
    if (ValuePtr == NULL || ResultPtr == NULL || Size == 0) {
        return -1;
    }

    EMConfig_t cfg = EMConfigExponentialDefault;
    if (ConfigPtr) {
        cfg = *ConfigPtr;
    }
    // First we must initialize the parameters.
    //  I will provide a function for this.  The first function will just initialize them
    //  randomly depending on the value vector.
    RandomInitExponentialEM(ValuePtr, Size, NumExponentials, ResultPtr);

    int mconv = 0;
    int pconv = 0;

    double* mcur = ResultPtr->means_init;
    double* mprev = ResultPtr->means_final;
    double* pcur = ResultPtr->probs_init;
    double* pprev = ResultPtr->probs_final;

    double merr = 0.0;
    double perr = 0.0;
    int iter = 1;

    if (cfg.verbose) {
        printf("INFO: Exponential Mixture Deconvolution by Expectation Maximization\n");
        printf("INFO:  Number of Exponentials involved in mixture - %d\n", NumExponentials);
        printf("INFO:  Number of Values observed total - %lu\n", Size);
        printf("INFO:  rtole - %f\n", cfg.rtole);
        printf("INFO:  maxiter - %d\n", cfg.maxiter);
    }

    // Likelihood matrix
    //     matrix
    //      +----Size-----
    //      |
    //   NumDist = (NumExponentials)
    //      |
    //      +-------------
    //
    double* lhall = NULL;
    lhall = (double *)malloc(sizeof(double) * Size * NumExponentials);
    memset(lhall, 0, sizeof(double) * Size * NumExponentials);

    double* tmpvec = NULL;
    tmpvec = (double *)malloc(sizeof(double) * Size);
    memset(tmpvec, 0, sizeof(double) * Size);

    while ((!mconv || !pconv) && iter <= cfg.maxiter) {
        if (cfg.verbose) {
            printf("INFO:  EM Iteration - %d | Rel. Error Mean = %e | Rel. Error Props = %e\n", iter, merr, perr);
        }

        // Get the Likelihoods
        double rowtotal = 0;

        for (int j = 0; j < NumExponentials; j++) {
            const double m = mprev[j];
            const double p = pprev[j];

            for (int i = 0; i < Size; i++) {
                double lh = GetExpoLH(ValuePtr[i], m) * p;

                if (lh == 0) {
                    lh = 0.000001;
                }
                lhall[j * Size + i] = lh;
            }
        }

        for (int i = 0; i < Size; i++) {
            double total = 0.0;

            for (int j = 0; j < NumExponentials; j++) {
                total += lhall[j * Size + i];
            }
            for (int j = 0; j < NumExponentials; j++) {
                const size_t offset = j * Size + i;
                lhall[offset] /= total;
            }
        }

        // Generate the new proportion estimates
        for(int j = 0; j < NumExponentials; j++) {
            pcur[j] = SumVectorD(&lhall[j * Size], Size);
            pcur[j] /= Size;
        }

        // Get the error estimate
        perr = 0;
        for (int j = 0; j < NumExponentials; j++) {
            perr += sqrt(pow(pcur[j]-pprev[j],2));
        }
        perr /= NumExponentials;


        pconv = (perr < cfg.rtole);

        // Generate the mean estimates
        double colsum;
        for (int j = 0; j < NumExponentials; j++) {
            MulVectorD(&lhall[j * Size], ValuePtr, tmpvec, Size);
            mcur[j] = SumVectorD(tmpvec, Size);
            colsum = SumVectorD(&lhall[j * Size], Size);
            mcur[j] /= colsum;
        }

        // Get the error estimate
        merr = 0;
        for (int j = 0; j < NumExponentials; j++) {
            merr += sqrt(pow(mcur[j]-mprev[j],2));
        }
        merr /= NumExponentials;

        mconv = (merr < cfg.rtole);

        // Switch cur and prev
        double *tmp = pprev;
        pprev = pcur;
        pcur = tmp;

        tmp = mprev;
        mprev = mcur;
        mcur = tmp;

        iter++;
    }

    ResultPtr->iterstaken = iter;

    ResultPtr->probs_init = pcur;
    ResultPtr->probs_final = pprev;

    ResultPtr->means_init = mcur;
    ResultPtr->means_final = mprev;


    free(tmpvec);
    free(lhall);

    return 0;

}

int RandomInitExponentialEM(const double* ValuePtr, size_t Size, int NumExponentials, EMResultExponential_t* ResultPtr)
{
    ResultPtr->numExponentials = NumExponentials;

    double minv = Min(ValuePtr, Size);
    double maxv = Max(ValuePtr, Size);
    double meanv = Mean(ValuePtr, Size);
    double varv = Var(ValuePtr, Size);

    size_t VecSize = sizeof(double) * NumExponentials;

    ResultPtr->means_init = (double *)malloc(VecSize);
    ResultPtr->probs_init = (double *)malloc(VecSize);

    ResultPtr->means_final = (double *)malloc(VecSize);
    ResultPtr->probs_final = (double *)malloc(VecSize);

    for (int i = 0; i < NumExponentials; i++) {
        // Initialize the means randomly as starting values in the range.
        int ri = rand() % 10000;
        double rd = ri / 10000.0;
        double rnmn = (maxv - minv) * rd + minv;
        ResultPtr->means_init[i] = rnmn;

        // Initialize the probabilities as uniform
        ResultPtr->probs_init[i] = 1.0 / NumExponentials;
    }

    memcpy(ResultPtr->means_final, ResultPtr->means_init, VecSize);
    memcpy(ResultPtr->probs_final, ResultPtr->probs_init, VecSize);

    return 0;
}

double GetExpoLH(double value, double mn)
{
    if (value < 0){
        return(0.0);
    } else {
        double LH = (1.0/mn) * pow(2.718281,-(1.0/mn)*value);
        return(LH);
    }
}

int ExpectationMaximization(const char* CompatMatrixPtr, size_t NumRows, size_t NumCols, const int* CountPtr, size_t NumCount, EMResult_t* ResultPtr, EMConfig_t* ConfigPtr)
{
    if (CompatMatrixPtr == NULL || CountPtr == NULL || ResultPtr == NULL) {
        return -1;
    }

    if (NumRows != NumCount) {
        return -2;
    }

    EMConfig_t cfg = EMConfigDefault;
    if (ConfigPtr) {
        cfg = *ConfigPtr;
    }

    const int NumTrans = NumCols;
    const int NumPattern = NumRows;

    double* abdninit = NULL; // Initialization vector for the abundance values
    double* abdn_new = NULL; // For storing the newly computed abundance values at a given stage
    double* abdn_old = NULL; // For containing the older abundance values.
    double* expected_counts = NULL; // For storing the intermediate expected counts as they are recomputed at each iteration of the EM algorithm.

    int totalreads = 0; // For storing the total number of reads in the comppatcounts struct.
    int totalcomp = 0; // for storing the total number of compatible transcripts for each read in the same structure.
    double abdn_total = 0; // for storing the total abundance on each row of the compatibility matrix.
    double relerr = 1000;

    unsigned long long start_ts = GetCurrentTimestamp();

    // allocate abundance vectors
    size_t vecsize = sizeof(double) * NumTrans;
    abdninit = (double *)malloc(vecsize);
    abdn_new = (double *)malloc(vecsize);
    abdn_old = (double *)malloc(vecsize);
    expected_counts = (double *)malloc(vecsize);

    //totalreads = SumVectorI(CountPtr, NumPattern); 
    // (09-2022: MT:  I had to implement these by hand to get this to work correctly)
    totalreads = 0; 
    for (int k = 0; k < NumPattern; k++){
        totalreads += CountPtr[k];
    }

    double tmp_counts = totalreads / NumTrans;
    double tmp_abdn = 1.0 / NumTrans;
    for (int i = 0; i < NumTrans; i++) {
        abdninit[i] = tmp_abdn;
        expected_counts[i] = tmp_counts;
    }

    memcpy(abdn_new, abdninit, vecsize); // initialize both the previous and current iteration abundances to the initialization values.
    memcpy(abdn_old, abdninit, vecsize);

    int num_iter = 0;

    while (relerr > cfg.rtole && num_iter < cfg.maxiter) {

        memset(expected_counts, 0, vecsize);

        for (int i = 0; i < NumPattern; i++){
            abdn_total = 0;
            totalcomp = 0;
            for (int j = 0; j < NumTrans; j++){
                const size_t offset = i * NumTrans + j;

                totalcomp += (int)(CompatMatrixPtr[offset] == '1');
                if (CompatMatrixPtr[offset] == '1') {
                    abdn_total += abdn_old[j];
                }
            }
            for (int j = 0; j < NumTrans; j++){
                const size_t offset = i * NumTrans + j;
                if (CompatMatrixPtr[offset] == '1') {
                    expected_counts[j] += ((abdn_old[j]/abdn_total) * CountPtr[i]);
                }
            }
        }

       // DivVectorValD(expected_counts, totalreads, abdn_new, NumTrans);
       // (09-2022: MT:  Again I had to implement this in order to avoid a segmentation fault for some reason)
       for (int k = 0; k < NumTrans; k++){
        abdn_new[k] = expected_counts[k]/totalreads;
       }

        //relerr = GetRelErrVectorD(abdn_new, abdn_old, NumTrans);
        // (09-2022: MT:  This was the last one that seemed to be problematic)
        relerr = 0;
        for (int k = 0; k < NumTrans; k++){
            relerr += (abdn_new[k]-abdn_old[k])*(abdn_new[k]-abdn_old[k]);
        }
        relerr /= NumTrans;
        relerr = sqrt(relerr);
        if (cfg.verbose) {
            printf("INFO: EM - Iteration %d Relative Error: %f\n", num_iter, relerr);
        }

        double *tmp = abdn_old;
        abdn_old = abdn_new;
        abdn_new = tmp;
        num_iter++;
    }

     if (cfg.verbose) {
         printf("INFO: EM - Ran for %d iteration\n", num_iter);
         unsigned long long end_ts = GetCurrentTimestamp();
         printf("INFO: EM - Took %llu microseconds to run\n", end_ts - start_ts);
     }

     ResultPtr->size = NumTrans;
     ResultPtr->values = abdn_old;

     free(abdn_new);
     free(abdninit);
     free(expected_counts);

    return 0;
}

// Utility Functions

#define FREE_IF(v) do { \
    if ((v)) {          \
        free((v));      \
        (v) = NULL;     \
    }                   \
} while(0)

void ReleaseEMResult(EMResult_t* ResultPtr)
{
    if (ResultPtr == NULL) {
        return;
    }

    FREE_IF(ResultPtr->values);
}

void ReleaseEMResultGaussian(EMResultGaussian_t* ResultPtr)
{
    if (ResultPtr == NULL) {
        return;
    }

    FREE_IF(ResultPtr->means_init);
    FREE_IF(ResultPtr->vars_init);
    FREE_IF(ResultPtr->probs_init);
    FREE_IF(ResultPtr->means_final);
    FREE_IF(ResultPtr->vars_final);
    FREE_IF(ResultPtr->probs_final);
}

void ReleaseEMResultExponential(EMResultExponential_t* ResultPtr)
{
    if (ResultPtr == NULL) {
        return;
    }
    FREE_IF(ResultPtr->probs_init);
    FREE_IF(ResultPtr->means_init);
    FREE_IF(ResultPtr->probs_final);
    FREE_IF(ResultPtr->means_final);
}

void GetEMDefaultConfig(EMConfig_t* ConfigPtr)
{
    if (ConfigPtr) {
        *ConfigPtr = EMConfigDefault;
    }
}
void GetEMGaussianDefaultConfig(EMConfig_t* ConfigPtr)
{
    if (ConfigPtr) {
        *ConfigPtr = EMConfigGaussianDefault;
    }
}
void GetEMExponentialDefaultConfig(EMConfig_t* ConfigPtr)
{
    if (ConfigPtr) {
        *ConfigPtr = EMConfigExponentialDefault;
    }
}
#undef FREE_IF

