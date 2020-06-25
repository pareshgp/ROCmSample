#include <iostream>
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include <rocblas.h>
#include <roctx.h>
#include <chrono>
#include<roctracer.h>
#include<roctracer_hsa.h>
#include<roctracer_hip.h>
#include<roctracer_kfd.h>

rocblas_operation transa = rocblas_operation_none;
rocblas_operation transb = rocblas_operation_transpose;

float fast_pseudo_rand(u_long *nextr) {
    *nextr = *nextr * 1103515245 + 12345;
    return static_cast<float>(static_cast<uint32_t>
                    ((*nextr / 65536) % 320000)) / 0.1234;
}

void hsa_domain_api_compatibility_check(uint32_t rt_major_ver) {
    uint32_t domain, op, kind;

    std::cout << "ROCTracer Major version " << rt_major_ver << std::endl;
    std::cout << "Checking HSA API Domain IP compatibility Start" << std::endl;
    domain = ACTIVITY_DOMAIN_HSA_API;
    op = 1;
    kind = 0;
    const char* op_string = roctracer_op_string(domain, op, kind);
    printf("%s op_string\n", op_string);
    std::cout << "Checking HSA API Domain IP compatibility Stop" << std::endl;
}

void hip_domain_api_compatibility_check(uint32_t rt_major_ver) {
    uint32_t domain, op, kind;

    std::cout << "ROCTracer Major version " << rt_major_ver << std::endl;
    std::cout << "Checking HIP API Domain IP compatibility Start" << std::endl;
    domain = ACTIVITY_DOMAIN_HIP_API;
    op = 1;
    kind = 0;
    const char* op_string = roctracer_op_string(domain, op, kind);
    printf("%s op_string\n", op_string);
    std::cout << "Checking HIP API Domain IP compatibility Stop" << std::endl;
}

void kfd_domain_api_compatibility_check(uint32_t rt_major_ver) {
    uint32_t domain, op, kind;

    std::cout << "ROCTracer Major version " << rt_major_ver << std::endl;
    std::cout << "Checking KFD API Domain IP compatibility Start" << std::endl;
    domain = ACTIVITY_DOMAIN_KFD_API;
    op = HSA_OP_ID_COPY;
    kind = 0;
    const char* op_string = roctracer_op_string(domain, op, kind);
    printf("%s op_string\n", op_string);
    std::cout << "Checking KFD API Domain IP compatibility Stop" << std::endl;
}

int main() {

    // GPU device index, i is iterator
    int gpu_device_index, i;
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

    uint64_t nextr = time(NULL);

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

    gpu_device_index = 0;
    // select GPU device & allocate memory
    if (hipSetDevice(gpu_device_index) != hipSuccess) {
        // cannot select the given GPU device
		std::cout << "cannot select the given GPU device ";
        return -1;
    } else {
        if (rocblas_create_handle(&blas_handle) == rocblas_status_success) {
            is_handle_init = true;
            /*if (rocblas_get_stream(blas_handle, &hip_stream)
                 != rocblas_status_success) {
			    std::cout << "rocblas hip_stream get failed" ;
                return -1;
            }*/
        } else {
			std::cout << "rocblas_create_handle failed" ;
            return -1;
        }
    }

    for (i = 0; i < size_a; ++i)
        ha[i] = fast_pseudo_rand(&nextr);

    for (i = 0; i < size_b; ++i)
        hb[i] = fast_pseudo_rand(&nextr);

    for (int i = 0; i < size_c; ++i)
        hc[i] = fast_pseudo_rand(&nextr);
	
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

    int numRepeats = 100;
    float alpha = 1.1, beta = 0.9;
    auto start_time = std::chrono::system_clock::now();
    for (int i = 0; i < numRepeats; ++i) {	
        rocblas_status stat = rocblas_sgemm(blas_handle, transa, transb,
                 m, n, k,
                 &alpha, da, m,
                 db, n, &beta,
                 dc, m);
        if (stat != rocblas_status_success){
            std::cout << "rocblas_sgemm failed" ;
            return -1;
        }
    }

    using namespace std::this_thread; // sleep_for, sleep_until
    using namespace std::chrono; // nanoseconds, system_clock, seconds

    sleep_for(nanoseconds(3000000));
    sleep_until(system_clock::now() + seconds(1));
    
    uint32_t roctracer_major_ver = roctracer_version_major();
    hsa_domain_api_compatibility_check(roctracer_major_ver);
    hip_domain_api_compatibility_check(roctracer_major_ver);
    kfd_domain_api_compatibility_check(roctracer_major_ver);


    hipDeviceSynchronize();
    auto end_time = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    std::cout << "Time to execute " << numRepeats << " kernels "
                   << diff.count() << " s\n";

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
