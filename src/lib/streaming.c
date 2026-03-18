/*
 * Copyright 2022-2026, Micah Thornton and Chanhee Park
 * Streaming EM — file-based chunked processing for datasets that don't fit in RAM.
 * License: GPL v3
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "streaming.h"
#include "distributions.h"

#define STREAM_PDF_FLOOR 1e-300

int UnmixStreaming(const char* filename, const StreamConfig* config,
                   MixtureResult* result)
{
    if (!filename || !config || !result) return -1;

    int k = config->num_components;
    int chunk_size = config->chunk_size > 0 ? config->chunk_size : 10000;
    int max_passes = config->max_passes > 0 ? config->max_passes : 10;
    double rtole = config->rtole > 0 ? config->rtole : 1e-5;
    double decay = config->eta_decay > 0 ? config->eta_decay : 0.6;
    DistFamily family = config->family;

    const DistFunctions* df = GetDistFunctions(family);
    if (!df) return -2;

    /* Allocate result */
    result->family = family;
    result->num_components = k;
    result->mixing_weights = (double*)malloc(sizeof(double) * k);
    result->params = (DistParams*)malloc(sizeof(DistParams) * k);
    if (!result->mixing_weights || !result->params) {
        free(result->mixing_weights); result->mixing_weights = NULL;
        free(result->params);         result->params = NULL;
        return -5;
    }

    /* First pass: count lines and compute global stats for initialization */
    FILE* fp = fopen(filename, "r");
    if (!fp) return -3;

    size_t total_n = 0;
    double global_sum = 0, global_sum2 = 0;
    double global_min = 1e30, global_max = -1e30;
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        double v = atof(line);
        global_sum += v;
        global_sum2 += v * v;
        if (v < global_min) global_min = v;
        if (v > global_max) global_max = v;
        total_n++;
    }
    fclose(fp);

    if (total_n == 0) return -4;

    double global_mean = global_sum / total_n;
    double global_var = global_sum2 / total_n - global_mean * global_mean;
    if (global_var < 1e-10) global_var = 1e-10;

    if (config->verbose)
        printf("  [stream] n=%zu  mean=%.4f  var=%.4f  range=[%.2f, %.2f]\n",
               total_n, global_mean, global_var, global_min, global_max);

    /* Initialize parameters: spread means evenly across data range */
    for (int j = 0; j < k; j++) {
        result->mixing_weights[j] = 1.0 / k;
        double frac = (k > 1) ? (double)j / (k - 1) : 0.5;
        double mu = global_min + frac * (global_max - global_min);
        result->params[j].p[0] = mu;
        if (df->num_params >= 2) result->params[j].p[1] = global_var / k;
        result->params[j].nparams = df->num_params;
    }

    /* Running sufficient statistics per component */
    double* suf_w = (double*)calloc(k, sizeof(double));
    double* suf_wx = (double*)calloc(k, sizeof(double));
    double* suf_wxx = (double*)calloc(k, sizeof(double));
    if (!suf_w || !suf_wx || !suf_wxx) {
        free(suf_w); free(suf_wx); free(suf_wxx);
        free(result->mixing_weights); result->mixing_weights = NULL;
        free(result->params);         result->params = NULL;
        return -5;
    }

    /* Init sufficient stats */
    for (int j = 0; j < k; j++) {
        suf_w[j] = 1.0 / k;
        suf_wx[j] = result->params[j].p[0] / k;
        suf_wxx[j] = (result->params[j].p[0] * result->params[j].p[0] +
                       (df->num_params >= 2 ? result->params[j].p[1] : 1.0)) / k;
    }

    /* Allocate chunk buffer */
    double* chunk = (double*)malloc(sizeof(double) * chunk_size);
    double* chunk_resp = (double*)malloc(sizeof(double) * k * chunk_size);
    double* chunk_w = (double*)malloc(sizeof(double) * chunk_size);
    if (!chunk || !chunk_resp || !chunk_w) {
        free(chunk); free(chunk_resp); free(chunk_w);
        free(suf_w); free(suf_wx); free(suf_wxx);
        free(result->mixing_weights); result->mixing_weights = NULL;
        free(result->params);         result->params = NULL;
        return -5;
    }

    double prev_ll = -1e30;
    int global_step = 0;

    /* Streaming EM passes */
    for (int pass = 0; pass < max_passes; pass++) {
        fp = fopen(filename, "r");
        if (!fp) break;

        double pass_ll = 0;
        size_t pass_n = 0;
        int chunk_idx = 0;

        while (1) {
            /* Read a chunk */
            int n_read = 0;
            while (n_read < chunk_size && fgets(line, sizeof(line), fp)) {
                if (line[0] == '#' || line[0] == '\n') continue;
                chunk[n_read++] = atof(line);
            }
            if (n_read == 0) break;

            double eta = pow(global_step + 2.0, -decay);
            global_step++;

            /* E-step on chunk */
            double chunk_ll = 0;
            for (int i = 0; i < n_read; i++) {
                double total = 0;
                for (int j = 0; j < k; j++) {
                    double lp = df->logpdf ?
                        df->logpdf(chunk[i], &result->params[j]) :
                        log(df->pdf(chunk[i], &result->params[j]) + 1e-300);
                    double p = result->mixing_weights[j] * exp(lp);
                    if (p < STREAM_PDF_FLOOR) p = STREAM_PDF_FLOOR;
                    chunk_resp[j * n_read + i] = p;
                    total += p;
                }
                for (int j = 0; j < k; j++) chunk_resp[j * n_read + i] /= total;
                chunk_ll += log(total);
            }
            pass_ll += chunk_ll;
            pass_n += n_read;

            /* Stochastic M-step: update sufficient statistics */
            for (int j = 0; j < k; j++) {
                double new_w = 0, new_wx = 0, new_wxx = 0;
                for (int i = 0; i < n_read; i++) {
                    double r = chunk_resp[j * n_read + i];
                    new_w += r;
                    new_wx += r * chunk[i];
                    new_wxx += r * chunk[i] * chunk[i];
                }
                new_w /= n_read;
                new_wx /= n_read;
                new_wxx /= n_read;

                suf_w[j] = (1 - eta) * suf_w[j] + eta * new_w;
                suf_wx[j] = (1 - eta) * suf_wx[j] + eta * new_wx;
                suf_wxx[j] = (1 - eta) * suf_wxx[j] + eta * new_wxx;
            }

            /* Reconstruct parameters from sufficient statistics */
            double wsum = 0;
            for (int j = 0; j < k; j++) wsum += suf_w[j];
            for (int j = 0; j < k; j++) {
                result->mixing_weights[j] = suf_w[j] / wsum;
                if (result->mixing_weights[j] < 1e-10) result->mixing_weights[j] = 1e-10;

                double mu = suf_wx[j] / (suf_w[j] > 1e-10 ? suf_w[j] : 1e-10);
                double var = suf_wxx[j] / (suf_w[j] > 1e-10 ? suf_w[j] : 1e-10) - mu * mu;
                if (var < 1e-10) var = 1e-10;

                if (family == DIST_GAUSSIAN) {
                    result->params[j].p[0] = mu;
                    result->params[j].p[1] = var;
                    result->params[j].nparams = 2;
                } else {
                    /* General: use chunk as weighted pseudo-data */
                    for (int i = 0; i < n_read; i++)
                        chunk_w[i] = chunk_resp[j * n_read + i];
                    DistParams old = result->params[j];
                    df->estimate(chunk, chunk_w, n_read, &result->params[j]);
                    for (int q = 0; q < result->params[j].nparams; q++) {
                        if (!isfinite(result->params[j].p[q]))
                            result->params[j].p[q] = old.p[q];
                        else
                            result->params[j].p[q] = (1-eta)*old.p[q] + eta*result->params[j].p[q];
                    }
                }
            }

            chunk_idx++;
        }
        fclose(fp);

        double avg_ll = pass_ll / (pass_n > 0 ? pass_n : 1);
        if (config->verbose)
            printf("  [stream] pass %d/%d  chunks=%d  avg_LL=%.6f  eta=%.4f\n",
                   pass + 1, max_passes, chunk_idx, avg_ll, pow(global_step + 1.0, -decay));

        /* Check convergence */
        if (pass > 0 && fabs(avg_ll - prev_ll) < rtole) {
            if (config->verbose) printf("  [stream] converged at pass %d\n", pass + 1);
            break;
        }
        prev_ll = avg_ll;
    }

    /* Final full-pass LL computation */
    fp = fopen(filename, "r");
    double final_ll = 0;
    if (fp) {
        while (fgets(line, sizeof(line), fp)) {
            if (line[0] == '#' || line[0] == '\n') continue;
            double x = atof(line);
            double total = 0;
            for (int j = 0; j < k; j++) {
                double lp = df->logpdf ?
                    df->logpdf(x, &result->params[j]) :
                    log(df->pdf(x, &result->params[j]) + 1e-300);
                total += result->mixing_weights[j] * exp(lp);
            }
            final_ll += log(total > 1e-300 ? total : 1e-300);
        }
        fclose(fp);
    }

    result->loglikelihood = final_ll;
    result->iterations = global_step;
    int nfree = k * (df->num_params + 1) - 1;
    result->bic = -2 * final_ll + nfree * log((double)total_n);
    result->aic = -2 * final_ll + 2 * nfree;

    free(suf_w); free(suf_wx); free(suf_wxx);
    free(chunk); free(chunk_resp); free(chunk_w);
    return 0;
}
