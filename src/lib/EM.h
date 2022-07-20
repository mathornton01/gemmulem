#ifndef __EM_H__
#define __EM_H__


#ifdef __cplusplus
extern "C"
{
#endif

struct EMConfig {
    int verbose;
    int maxiter;
    double rtole;
};
typedef struct EMConfig EMConfig_t;

struct EMResult {
    // Array of expectations
    size_t size;
    double *values;
};
typedef struct EMResult EMResult_t;

struct EMResultGaussian {
    int numGaussians;
    int iterstaken;
    double* means_init;
    double* vars_init;
    double* probs_init;
    double* means_final;
    double* vars_final;
    double* probs_final;
};
typedef struct EMResultGaussian EMResultGaussian_t;

struct EMResultExponential {
    int numExponentials;
    int iterstaken;
    double* means_init;
    double* probs_init;
    double* means_final;
    double* probs_final;
};
typedef struct EMResultExponential EMResultExponential_t;

/**
 * Run ExpectationMaximization
 *
 * @param CompatMatrixPtr
 * @param ResultPtr
 * @param ConfigPtr
 * @return
 */
int ExpectationMaximization(const char* CompatMatrixPtr, size_t NumRows, size_t NumCols, const int* CountPtr, size_t NumCount, EMResult_t* ResultPtr, EMConfig_t* ConfigPtr);


// Function headers 
/**
 * Run EM algorithms for Gaussian
 *
 * @param ValuePtr
 * @param Size
 * @param NumGaussians
 * @param ResultPtr
 * @param ConfigPtr
 * @return
 */
int UnmixGaussians(const double* ValuePtr, size_t Size, int NumGaussians, EMResultGaussian_t* ResultPtr, EMConfig_t* ConfigPtr);

//TODO: Implement K-Means initialization procedure 
//struct gaussianemresults kmeansinitgem(std::vector<double> values, int numGaussians);

// Function headers 

/**
 * Run EM algorithms for Exponential
 *
 * @param ValuePtr
 * @param Size
 * @param NumExponentials
 * @param ResultPtr
 * @param ConfigPtr
 * @return
 */
int UnmixExponentials(const double* ValuePtr, size_t Size, int NumExponentials, EMResultExponential_t* ResultPtr, EMConfig_t* ConfigPtr);


//TODO: Implement K-Means initialization procedure 
//struct exponentialEMResults kmeansiniteem(vector<double> values, int numExponentials);



// Utility functions
/**
 * Release resources used in EMResult_t
 * @param ResultPtr
 *
 */
void ReleaseEMResult(EMResult_t* ResultPtr);

/**
 * Release resources used in EMResultGaussian_t
 * @param ResultPtr
 */
void ReleaseEMResultGaussian(EMResultGaussian_t* ResultPtr);

/**
 * Release resources used in EMResultExponential_t
 * @param ResultPtr
 */
void ReleaseEMResultExponential(EMResultExponential_t* ResultPtr);


/**
 * Get default configuration
 * @param ConfigPtr
 */
void GetEMDefaultConfig(EMConfig_t* ConfigPtr);
void GetEMGaussianDefaultConfig(EMConfig_t* ConfigPtr);
void GetEMExponentialDefaultConfig(EMConfig_t* ConfigPtr);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif /* __EM_H__ */
