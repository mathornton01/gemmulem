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

    size_t NumPattern = LENGTH(compats);
    SEXP StrItem = STRING_ELT(compats, 0);
    const char* PatternItem = CHAR(asChar(StrItem));
    size_t NumCategory = strlen(PatternItem);

    // alloc for CompatMatrixPtr
    char* CompatMatrixPtr = (char*)malloc(NumCategory * NumPattern);

    for(int i = 0; i < NumPattern; i++) {
        const char* ptr = CHAR(asChar(STRING_ELT(compats, i)));
        if (ptr != NULL) {
            memcpy(CompatMatrixPtr + i * NumCategory, ptr, NumCategory);
        }
    }

    const int* CountPtr = INTEGER(counts);
    size_t NumCount = LENGTH(counts);

    EMResult_t Results;
    ExpectationMaximization(
            CompatMatrixPtr,
            NumPattern, /* NumRows */
            NumCategory, /* NumCols */
            CountPtr,
            NumCount,
            &Results,
            NULL /* Default config */
            );

    SEXP ResultVec;
    int ResultSize = Results.size;

    PROTECT(ResultVec = allocVector(REALSXP, ResultSize));
    double* ResultPtr = REAL(ResultVec);
    memcpy(ResultPtr, Results.values, sizeof(double) * ResultSize);
    ReleaseEMResult(&Results);
    UNPROTECT(1);

    free(CompatMatrixPtr);
    return ResultVec;
}

SEXP r_unmixgaussians(SEXP values, SEXP num_dist) 
{
    int Size = LENGTH(values);
    int NumDist = asInteger(num_dist);

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
