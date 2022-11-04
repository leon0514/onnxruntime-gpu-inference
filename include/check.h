#ifndef CHECK_H__
#define CHECK_H__
#include <cuda_runtime.h>
#include <stdio.h>
#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line);
#endif