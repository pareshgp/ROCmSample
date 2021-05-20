#include <iostream>
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include <chrono>
#include <cstdlib>
#include <unistd.h>


float fast_pseudo_rand(u_long *nextr) {
    *nextr = *nextr * 1103515245 + 12345;
    return static_cast<float>(static_cast<uint32_t>
                    ((*nextr / 65536) % 320000)) / 0.1234;
}


int main(int argc, char **argv) {
    

    int numRepeats = 10;
    if (argc > 1) {
        if (!strcmp("-time", argv[1]))
        {
            if(argv[2]) {
                    numRepeats  = atoi(argv[2]);
	    }
        } else {
            numRepeats = 10;
        }
    }
	
    // GPU device index, i is iterator
    int gpu_device_index, i;
    // matrix size m

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
    int size_a;
    int size_b;
    int size_c;

    uint64_t nextr = time(NULL);

	// HIP API stream - used to query for GEMM completion
    hipStream_t hip_stream;

    gpu_device_index = 0;
    // select GPU device & allocate memory
    if (hipSetDevice(gpu_device_index) != hipSuccess) {
        // cannot select the given GPU device
		std::cout << "cannot select the given GPU device ";
        return -1;
    }

    auto start_time = std::chrono::system_clock::now();
    for (int iter = 0; iter < numRepeats; ++iter) {	
    size_a = 10000 * (iter+1);
    size_b = 10000 * (iter+1);
    size_c = 10000 * (iter+1);

	// allocate host matrix memory
	try {
        ha = new float[size_a];
        hb = new float[size_b];
        hc = new float[size_c];


    } catch (std::bad_alloc&) {
		std::cout << "Memory allocation failed for host matrix ";
        return -1;
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
    sleep(1);
    }
    auto end_time = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    std::cout << "Time to execute " << numRepeats << " hipMemCpy "
                   << diff.count() << " s\n";
return 0;
}
