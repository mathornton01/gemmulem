
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "distributions.h"

static double wall_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char* argv[]) {
    if (argc < 3) { fprintf(stderr, "Usage: gem_bench <datafile> <family> [k]\n"); return 1; }

    /* Read data */
    FILE* f = fopen(argv[1], "r");
    if (!f) { perror(argv[1]); return 1; }
    int n = 0; double buf[100000]; double v;
    while (fscanf(f, "%lf", &v) == 1 && n < 100000) buf[n++] = v;
    fclose(f);

    /* Family */
    DistFamily fam = DIST_GAUSSIAN;
    if (strcmp(argv[2], "gaussian") == 0)  fam = DIST_GAUSSIAN;
    else if (strcmp(argv[2], "student_t") == 0) fam = DIST_STUDENT_T;
    else if (strcmp(argv[2], "laplace") == 0)  fam = DIST_LAPLACE;
    else if (strcmp(argv[2], "gamma") == 0)    fam = DIST_GAMMA;
    else if (strcmp(argv[2], "lognormal") == 0) fam = DIST_LOGNORMAL;
    else if (strcmp(argv[2], "weibull") == 0)  fam = DIST_WEIBULL;
    else if (strcmp(argv[2], "pearson") == 0)  fam = DIST_PEARSON;
    else if (strcmp(argv[2], "auto") == 0)     fam = (DistFamily)-1;

    int k = (argc >= 4) ? atoi(argv[3]) : 2;

    double t0 = wall_ms();
    MixtureResult result;
    int rc;

    if ((int)fam == -1) {
        /* Auto mode */
        ModelSelectResult msr;
        rc = SelectBestMixture(buf, n, NULL, 0, 1, k, 300, 1e-5, 0, &msr);
        if (rc == 0) {
            printf("{\"family\":\"%s\",\"k\":%d,\"ll\":%.4f,\"bic\":%.4f,\"iters\":%d,\"ms\":%.2f}\n",
                   GetDistName(msr.best_family), msr.best_k,
                   msr.candidates[0].loglikelihood,
                   msr.best_bic, 0, wall_ms()-t0);
            ReleaseModelSelectResult(&msr);
        }
        return rc;
    }

    rc = UnmixGeneric(buf, n, fam, k, 300, 1e-5, 0, &result);
    double ms = wall_ms() - t0;
    if (rc == 0) {
        printf("{\"family\":\"%s\",\"k\":%d,\"ll\":%.4f,\"bic\":%.4f,\"iters\":%d,\"ms\":%.2f,\"weights\":[",
               GetDistName(fam), k, result.loglikelihood, result.bic, result.iterations, ms);
        for (int j=0;j<k;j++) printf("%.4f%s",result.mixing_weights[j],j<k-1?",":"");
        printf("],\"params\":[");
        for (int j=0;j<k;j++) {
            printf("[");
            for (int q=0;q<result.params[j].nparams;q++)
                printf("%.4f%s",result.params[j].p[q],q<result.params[j].nparams-1?",":"");
            printf("]%s",j<k-1?",":"");
        }
        printf("]}\n");
        ReleaseMixtureResult(&result);
    } else {
        printf("{\"error\":%d}\n", rc);
    }
    return rc;
}
