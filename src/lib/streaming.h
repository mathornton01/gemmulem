/*
 * Copyright 2022-2026, Micah Thornton and Chanhee Park
 * Streaming EM — processes data from file in chunks without loading all into RAM.
 * License: GPL v3
 */
#ifndef STREAMING_H
#define STREAMING_H

#include <stddef.h>
#include "distributions.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int num_components;
    int chunk_size;        /* rows per chunk */
    int max_passes;        /* passes over the file */
    double rtole;          /* convergence tolerance */
    int verbose;
    DistFamily family;
    double eta_decay;      /* step-size decay: eta = (pass*n_chunks + chunk + 2)^(-decay) */
} StreamConfig;

/**
 * Stream EM: fits a mixture model by reading data from a file in chunks.
 * Never loads more than chunk_size values into RAM at once.
 *
 * @param filename   Path to data file (one value per line)
 * @param config     Streaming configuration
 * @param result     Output mixture result
 * @return 0 on success, -1 on error
 */
int UnmixStreaming(const char* filename, const StreamConfig* config,
                   MixtureResult* result);

#ifdef __cplusplus
}
#endif

#endif /* STREAMING_H */
