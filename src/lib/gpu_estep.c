/*
 * Copyright 2022-2026, Micah Thornton and Chanhee Park
 * GPU-accelerated E-step via OpenCL.
 * License: GPL v3
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gpu_estep.h"

#ifdef GEMMULEM_OPENCL
#include <CL/cl.h>

/* ════════════════════════════════════════════════════════════════════
 * OpenCL kernel source — Gaussian E-step
 * Each work item computes responsibilities for one data point.
 * ════════════════════════════════════════════════════════════════════ */
static const char* KERNEL_GAUSSIAN_ESTEP = "\n\
__kernel void gaussian_estep(\n\
    __global const float* data,\n\
    __global const float* log_weights,\n\
    __global const float* means,\n\
    __global const float* variances,\n\
    int k,\n\
    __global float* resp,\n\
    __global float* ll_partial\n\
) {\n\
    int i = get_global_id(0);\n\
    int n = get_global_size(0);\n\
    float xi = data[i];\n\
\n\
    /* Compute log-probabilities for each component */\n\
    float lps[64];\n\
    float max_lp = -1e30f;\n\
    for (int j = 0; j < k; j++) {\n\
        float mu = means[j], var = variances[j];\n\
        float lp = log_weights[j]\n\
                   - 0.5f * log(2.0f * 3.14159265f * var)\n\
                   - 0.5f * (xi - mu) * (xi - mu) / var;\n\
        lps[j] = lp;\n\
        if (lp > max_lp) max_lp = lp;\n\
    }\n\
\n\
    /* Log-sum-exp normalization */\n\
    float total = 0.0f;\n\
    for (int j = 0; j < k; j++) {\n\
        float v = exp(lps[j] - max_lp);\n\
        resp[j * n + i] = v;\n\
        total += v;\n\
    }\n\
    for (int j = 0; j < k; j++)\n\
        resp[j * n + i] /= total;\n\
\n\
    ll_partial[i] = max_lp + log(total);\n\
}\n";

struct GpuContext {
    cl_platform_id platform;
    cl_device_id   device;
    cl_context     context;
    cl_command_queue queue;
    cl_program     program;
    cl_kernel      kernel_gaussian;
    int            initialized;
};

GpuContext* gpu_init(int prefer_gpu) {
    GpuContext* ctx = (GpuContext*)calloc(1, sizeof(GpuContext));

    cl_uint num_platforms = 0;
    if (clGetPlatformIDs(0, NULL, &num_platforms) != CL_SUCCESS || num_platforms == 0) {
        free(ctx);
        return NULL;
    }

    cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
    clGetPlatformIDs(num_platforms, platforms, NULL);

    cl_device_id best_device = NULL;
    cl_platform_id best_platform = NULL;
    cl_device_type target = (prefer_gpu >= 1) ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_ALL;

    for (cl_uint p = 0; p < num_platforms; p++) {
        cl_uint num_devices = 0;
        clGetDeviceIDs(platforms[p], target, 0, NULL, &num_devices);
        if (num_devices == 0) continue;

        cl_device_id* devices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
        clGetDeviceIDs(platforms[p], target, num_devices, devices, NULL);

        /* Pick device with most compute units */
        for (cl_uint d = 0; d < num_devices; d++) {
            cl_uint cu;
            clGetDeviceInfo(devices[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL);
            if (best_device == NULL) {
                best_device = devices[d];
                best_platform = platforms[p];
            } else {
                cl_uint best_cu;
                clGetDeviceInfo(best_device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(best_cu), &best_cu, NULL);
                if (cu > best_cu) {
                    best_device = devices[d];
                    best_platform = platforms[p];
                }
            }
        }
        free(devices);
    }
    free(platforms);

    if (best_device == NULL) { free(ctx); return NULL; }

    /* Print selected device */
    char device_name[256] = {0};
    clGetDeviceInfo(best_device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    fprintf(stderr, "[GPU] Using device: %s\n", device_name);

    cl_int err;
    ctx->platform = best_platform;
    ctx->device = best_device;
    ctx->context = clCreateContext(NULL, 1, &best_device, NULL, NULL, &err);
    if (err != CL_SUCCESS) { free(ctx); return NULL; }

    ctx->queue = clCreateCommandQueue(ctx->context, best_device, 0, &err);
    if (err != CL_SUCCESS) { clReleaseContext(ctx->context); free(ctx); return NULL; }

    /* Compile kernel */
    ctx->program = clCreateProgramWithSource(ctx->context, 1, &KERNEL_GAUSSIAN_ESTEP, NULL, &err);
    if (err != CL_SUCCESS) { clReleaseCommandQueue(ctx->queue); clReleaseContext(ctx->context); free(ctx); return NULL; }

    err = clBuildProgram(ctx->program, 1, &best_device, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        char build_log[4096];
        clGetProgramBuildInfo(ctx->program, best_device, CL_PROGRAM_BUILD_LOG,
                              sizeof(build_log), build_log, NULL);
        fprintf(stderr, "[GPU] Build error: %s\n", build_log);
        clReleaseProgram(ctx->program);
        clReleaseCommandQueue(ctx->queue);
        clReleaseContext(ctx->context);
        free(ctx);
        return NULL;
    }

    ctx->kernel_gaussian = clCreateKernel(ctx->program, "gaussian_estep", &err);
    if (err != CL_SUCCESS) {
        clReleaseProgram(ctx->program);
        clReleaseCommandQueue(ctx->queue);
        clReleaseContext(ctx->context);
        free(ctx);
        return NULL;
    }

    ctx->initialized = 1;
    return ctx;
}

void gpu_destroy(GpuContext* ctx) {
    if (!ctx) return;
    if (ctx->initialized) {
        clReleaseKernel(ctx->kernel_gaussian);
        clReleaseProgram(ctx->program);
        clReleaseCommandQueue(ctx->queue);
        clReleaseContext(ctx->context);
    }
    free(ctx);
}

int gpu_estep_gaussian(GpuContext* ctx,
                       const float* data, int n,
                       const float* log_weights,
                       const float* means,
                       const float* variances,
                       int k,
                       float* resp,
                       double* ll_out)
{
    if (!ctx || !ctx->initialized) return -1;

    cl_int err;
    size_t data_size   = sizeof(float) * n;
    size_t param_size  = sizeof(float) * k;
    size_t resp_size   = sizeof(float) * n * k;
    size_t ll_size     = sizeof(float) * n;

    /* Allocate GPU buffers */
    cl_mem buf_data  = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, data_size,   (void*)data,        &err); if (err) return -1;
    cl_mem buf_lw    = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, param_size,  (void*)log_weights, &err); if (err) goto cleanup1;
    cl_mem buf_mu    = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, param_size,  (void*)means,       &err); if (err) goto cleanup2;
    cl_mem buf_var   = clCreateBuffer(ctx->context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, param_size,  (void*)variances,   &err); if (err) goto cleanup3;
    cl_mem buf_resp  = clCreateBuffer(ctx->context, CL_MEM_WRITE_ONLY,                         resp_size,   NULL,               &err); if (err) goto cleanup4;
    cl_mem buf_ll    = clCreateBuffer(ctx->context, CL_MEM_WRITE_ONLY,                         ll_size,     NULL,               &err); if (err) goto cleanup5;

    /* Set kernel args */
    clSetKernelArg(ctx->kernel_gaussian, 0, sizeof(cl_mem), &buf_data);
    clSetKernelArg(ctx->kernel_gaussian, 1, sizeof(cl_mem), &buf_lw);
    clSetKernelArg(ctx->kernel_gaussian, 2, sizeof(cl_mem), &buf_mu);
    clSetKernelArg(ctx->kernel_gaussian, 3, sizeof(cl_mem), &buf_var);
    clSetKernelArg(ctx->kernel_gaussian, 4, sizeof(cl_int), &k);
    clSetKernelArg(ctx->kernel_gaussian, 5, sizeof(cl_mem), &buf_resp);
    clSetKernelArg(ctx->kernel_gaussian, 6, sizeof(cl_mem), &buf_ll);

    /* Enqueue kernel: one work item per data point */
    size_t global_size = n;
    size_t local_size  = 256;
    if (global_size % local_size) local_size = 1;
    err = clEnqueueNDRangeKernel(ctx->queue, ctx->kernel_gaussian, 1,
                                 NULL, &global_size, &local_size, 0, NULL, NULL);
    if (err) goto cleanup6;

    /* Read back results */
    float* resp_f = (float*)malloc(resp_size);
    float* ll_f   = (float*)malloc(ll_size);
    clEnqueueReadBuffer(ctx->queue, buf_resp, CL_TRUE, 0, resp_size, resp_f, 0, NULL, NULL);
    clEnqueueReadBuffer(ctx->queue, buf_ll,   CL_TRUE, 0, ll_size,   ll_f,   0, NULL, NULL);

    /* Copy to output */
    memcpy(resp, resp_f, resp_size);
    *ll_out = 0;
    for (int i = 0; i < n; i++) *ll_out += ll_f[i];

    free(resp_f); free(ll_f);

cleanup6: clReleaseMemObject(buf_ll);
cleanup5: clReleaseMemObject(buf_resp);
cleanup4: clReleaseMemObject(buf_var);
cleanup3: clReleaseMemObject(buf_mu);
cleanup2: clReleaseMemObject(buf_lw);
cleanup1: clReleaseMemObject(buf_data);
    return err ? -1 : 0;
}

int gpu_available(void) {
    cl_uint n = 0;
    return (clGetPlatformIDs(0, NULL, &n) == CL_SUCCESS && n > 0);
}

#else /* No OpenCL — stub implementations */

struct GpuContext { int dummy; };

GpuContext* gpu_init(int prefer_gpu) {
    (void)prefer_gpu;
    return NULL;  /* No GPU support compiled in */
}
void gpu_destroy(GpuContext* ctx) { (void)ctx; }
int gpu_estep_gaussian(GpuContext* ctx, const float* data, int n,
                       const float* lw, const float* mu, const float* var,
                       int k, float* resp, double* ll_out) {
    (void)ctx; (void)data; (void)n; (void)lw; (void)mu; (void)var;
    (void)k; (void)resp; (void)ll_out;
    return -1;
}
int gpu_available(void) { return 0; }

#endif /* GEMMULEM_OPENCL */
