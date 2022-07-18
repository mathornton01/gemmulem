/**
 * R callable function
 *
 *
 */

#include <R.h>
#include <Rdefines.h>

#include <EM.h>

SEXP r_expectationmaximization(SEXP compats, SEXP counts) 
{
    SEXP ResultVec;

    PROTECT(ResultVec = allocVector(INTSXP, 1));
    INTEGER(ResultVec)[0] = 0;
    UNPROTECT(1);

    return ResultVec;
}

SEXP r_unmixgaussians(SEXP values, SEXP num_dist) 
{
    int Size = LENGTH(values);
    int NumDist = asInteger(num_dist);

    printf("Size=%d, NumDist=%d\n", Size, NumDist);

    EMResultGaussian_t Results;
    UnmixGaussians(REAL(values), Size, NumDist, &Results, NULL);

    int ResultSize = NumDist * 3; /* means, vars, probs */

    SEXP ResultVec;

    PROTECT(ResultVec = allocVector(REALSXP, ResultSize));

    double* ResultPtr = REAL(ResultVec);
    memcpy(ResultPtr, Results.means_final, sizeof(double) * Results.numGaussians);
    ResultPtr += Results.numGaussians;

    memcpy(ResultPtr, Results.vars_final, sizeof(double) * Results.numGaussians);
    ResultPtr += Results.numGaussians;

    memcpy(ResultPtr, Results.probs_final, sizeof(double) * Results.numGaussians);

    ReleaseEMResultGaussian(&Results);

    UNPROTECT(1);

    return ResultVec;
}

SEXP r_unmixexponentials(SEXP values, SEXP num_dist) 
{
    int Size = LENGTH(values);
    int NumDist = asInteger(num_dist);

    printf("Size=%d, NumDist=%d\n", Size, NumDist);

    EMResultExponential_t Results;
    UnmixExponentials(REAL(values), Size, NumDist, &Results, NULL);

    int ResultSize = NumDist * 2; /* means, probs */

    SEXP ResultVec;

    PROTECT(ResultVec = allocVector(REALSXP, ResultSize));

    double* ResultPtr = REAL(ResultVec);
    memcpy(ResultPtr, Results.means_final, sizeof(double) * Results.numExponentials);
    ResultPtr += Results.numExponentials;

    memcpy(ResultPtr, Results.probs_final, sizeof(double) * Results.numExponentials);

    ReleaseEMResultExponential(&Results);

    UNPROTECT(1);

    return ResultVec;
}
