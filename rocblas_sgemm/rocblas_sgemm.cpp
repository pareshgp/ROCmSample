#include <iostream>
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include <rocblas.h>
#include <roctx.h>

rocblas_operation transa = rocblas_operation_none;
rocblas_operation transb = rocblas_operation_transpose;

int main() {

    // GPU device index
    int gpu_device_index;
    // matrix size m
    rocblas_int m;
    // matrix size n
    rocblas_int n;
    // matrix size k
    rocblas_int k;
    // amount of memory to allocate for the matrix
    rocblas_int size_a;
    // amount of memory to allocate for the matrix
    rocblas_int size_b;
    // amount of memory to allocate for the matrix
    rocblas_int size_c;

    //SGEMM DECLARAION
    // pointer to device (GPU) memory
    float *da = NULL;
    // pointer to device (GPU) memory
    float *db = NULL;
    // pointer to device (GPU) memory
    float *dc = NULL;
    // pointer to host memorycd 
    float *ha = NULL;
    // pointer to host memory
    float *hb = NULL;
    // pointer to host memory
    float *hc = NULL;

	// HIP API stream - used to query for GEMM completion
    hipStream_t hip_stream;
    // rocBlas related handle
    rocblas_handle blas_handle;
    // TRUE is rocBlas handle was successfully initialized
    bool is_handle_init = false;

    m = n = k = 5760;
	
    size_a = k * m;
    size_b = k * n;
    size_c = n * m;

	// allocate host matrix memory
	try {
        ha = new float[size_a];
        hb = new float[size_b];
        hc = new float[size_c];


    } catch (std::bad_alloc&) {
		std::cout << "Memory allocation failed for host matrix ";
        return -1;
    }

	
	// allocates memory (for matrix multiplication) on the selected GPU
    if (hipMalloc(&da, size_a * sizeof(float)) != hipSuccess) {
		std::cout << "hipMalloc failed to allocate memory for da ";
        return -1;
	}
    if (hipMalloc(&db, size_b * sizeof(float)) != hipSuccess) {
		std::cout << "hipMalloc failed to allocate memory for da ";
        return -1;
	}
    if (hipMalloc(&dc, size_c * sizeof(float)) != hipSuccess) {
		std::cout << "hipMalloc failed to allocate memory for da ";
        return -1;
	}

    // select GPU device & allocate memory
    if (hipSetDevice(gpu_device_index) != hipSuccess) {
        // cannot select the given GPU device
		std::cout << "cannot select the given GPU device ";
        return -1;
    } else {
        if (rocblas_create_handle(&blas_handle) == rocblas_status_success) {
            is_handle_init = true;
            if (rocblas_get_stream(blas_handle, &hip_stream)
                 != rocblas_status_success)
			std::cout << "rocblas hip_stream get failed" ;
                return -1;		
        } else {
			std::cout << "rocblas_create_handle failed" ;
            return -1;
        }
    }

    roctxMark("before hipMemcpy");
    roctxRangePush("hipMemcpy");	
    // copy from Host to device start
    if (da) {
        if (hipMemcpy(da, ha, sizeof(float) * size_a, hipMemcpyHostToDevice)
            != hipSuccess) {
            std::cout << "hipMemcpyHostToDevice failed" ;
            return -1;
        }
    }

    if (db) {
        if (hipMemcpy(db, hb, sizeof(float) * size_b, hipMemcpyHostToDevice)
            != hipSuccess) {
            std::cout << "hipMemcpyHostToDevice failed" ;
            return -1;
        }
    }

    if (dc) {
        if (hipMemcpy(dc, hc, sizeof(float) * size_c, hipMemcpyHostToDevice)
            != hipSuccess) {
            std::cout << "hipMemcpyHostToDevice failed" ;
            return -1;
        }
    }
    roctxMark("after hipMemcpy");
    roctxRangePop();


    float alpha = 1.1, beta = 0.9;
	
    if (rocblas_sgemm(blas_handle, transa, transb,
                 m, n, k,
                 &alpha, da, m,
                 db, n, &beta,
                 dc, m) != rocblas_status_success) {
        std::cout << "hipMemcpyHostToDevice failed" ;
        return -1;
    }

    // releases the host matrix memory	
    if (ha)
        delete []ha;
    if (hb)
        delete []hb;
    if (hc)
        delete []hc;

	// 
	if (da)
        hipFree(da);
    if (db)
        hipFree(db);
    if (dc)
        hipFree(dc);

	if (is_handle_init)
        rocblas_destroy_handle(blas_handle);

return 0;
}
