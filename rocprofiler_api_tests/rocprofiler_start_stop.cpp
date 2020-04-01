#include <iostream>
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include <rocblas.h>
#include <hsa.h>
#include <string.h>
#include <unistd.h>
#include <iostream>
#include <rocprofiler.h>
#include "hsa_rsrc_factory.h"

#define TEST_STATUS(cond)                                                                          \
  {                                                                                                \
    if (!(cond)) {                                                                                 \
      std::cerr << "Test error at " << __FILE__ << ", line " << __LINE__ << std::endl;             \
      const char* message;                                                                         \
      rocprofiler_error_string(&message);                                                          \
      std::cerr << "ERROR: " << message << std::endl;                                              \
      exit(-1);                                                                                    \
    }                                                                                              \
  }


#define TEST_ASSERT(cond)                                                                          \
  {                                                                                                \
    if (!(cond)) {                                                                                 \
      std::cerr << "Assert failed(" << #cond << ") at " << __FILE__ << ", line " << __LINE__       \
                << std::endl;                                                                      \
      exit(-1);                                                                                    \
    }                                                                                              \
  }

rocblas_operation transa = rocblas_operation_none;
rocblas_operation transb = rocblas_operation_transpose;

float fast_pseudo_rand(u_long *nextr) {
    *nextr = *nextr * 1103515245 + 12345;
    return static_cast<float>(static_cast<uint32_t>
                    ((*nextr / 65536) % 320000)) / 0.1234;
}

void print_features(rocprofiler_feature_t* feature, uint32_t feature_count) {
    for (rocprofiler_feature_t* p = feature; p < feature + feature_count; ++p) {
      std::cout << (p - feature) << ": " << p->name;
      switch (p->data.kind) {
        case ROCPROFILER_DATA_KIND_INT64:
          std::cout << std::dec << " result64 (" << p->data.result_int64 << ")" << std::endl;
          break;
        case ROCPROFILER_DATA_KIND_BYTES: {
          const char* ptr = reinterpret_cast<const char*>(p->data.result_bytes.ptr);
          uint64_t size = 0;
          for (unsigned i = 0; i < p->data.result_bytes.instance_count; ++i) {
            size = *reinterpret_cast<const uint64_t*>(ptr);
            const char* data = ptr + sizeof(size);
            std::cout << std::endl;
            std::cout << std::hex << "  data (" << (void*)data << ")" << std::endl;
            std::cout << std::dec << "  size (" << size << ")" << std::endl;
            ptr = data + size;
          }
          break;
        }
        default:
          std::cout << "result kind (" << p->data.kind << ")" << std::endl;
          TEST_ASSERT(false);
      }
    }
}

void read_features(uint32_t n, rocprofiler_t* context, rocprofiler_feature_t* feature, const unsigned feature_count) {
    std::cout << "read features" << std::endl;
    hsa_status_t status = rocprofiler_read(context, n);
    TEST_STATUS(status == HSA_STATUS_SUCCESS);
    std::cout << "read issue" << std::endl;
    status = rocprofiler_get_data(context, n);
    TEST_STATUS(status == HSA_STATUS_SUCCESS);
    status = rocprofiler_get_metrics(context);
    TEST_STATUS(status == HSA_STATUS_SUCCESS);
    print_features(feature, feature_count);
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

    // HSA status
    hsa_status_t status = HSA_STATUS_ERROR;
    // HIP API stream - used to query for GEMM completion
    hipStream_t hip_stream;
    // rocBlas related handle
    rocblas_handle blas_handle;
    // TRUE is rocBlas handle was successfully initialized
    bool is_handle_init = false;

    // Profiling context
    rocprofiler_t* context = NULL;
    // Profiling properties
    rocprofiler_properties_t properties;

    // Profiling feature objects
    const unsigned feature_count = 9;
    rocprofiler_feature_t feature[feature_count];
    // PMC events
    memset(feature, 0, sizeof(feature));
    feature[0].kind = ROCPROFILER_FEATURE_KIND_METRIC;
    feature[0].name = "GRBM_COUNT";
    feature[1].kind = ROCPROFILER_FEATURE_KIND_METRIC;
    feature[1].name = "GRBM_GUI_ACTIVE";
    feature[2].kind = ROCPROFILER_FEATURE_KIND_METRIC;
    feature[2].name = "GPUBusy";
    feature[3].kind = ROCPROFILER_FEATURE_KIND_METRIC;
    feature[3].name = "SQ_WAVES";
    feature[4].kind = ROCPROFILER_FEATURE_KIND_METRIC;
    feature[4].name = "SQ_INSTS_VALU";
    feature[5].kind = ROCPROFILER_FEATURE_KIND_METRIC;
    feature[5].name = "VALUInsts";
    feature[6].kind = ROCPROFILER_FEATURE_KIND_METRIC;
    feature[6].name = "TCC_HIT_sum";
    feature[7].kind = ROCPROFILER_FEATURE_KIND_METRIC;
    feature[7].name = "TCC_MISS_sum";
    feature[8].kind = ROCPROFILER_FEATURE_KIND_METRIC;
    feature[8].name = "WRITE_SIZE";

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

    // Instantiate HSA resources
    HsaRsrcFactory::Create();
  
    // Getting GPU device info
    const AgentInfo* agent_info = NULL;
    if (HsaRsrcFactory::Instance().GetGpuAgentInfo(0, &agent_info) == false) abort();
  
    // Creating the queues pool
    const unsigned queue_count = 16;
    hsa_queue_t* queue[queue_count];
    for (unsigned queue_ind = 0; queue_ind < queue_count; ++queue_ind) {
      if (HsaRsrcFactory::Instance().CreateQueue(agent_info, 128, &queue[queue_ind]) == false) abort();
    }
    hsa_queue_t* prof_queue = queue[0];
  
    // Creating profiling context
    properties = {};
    properties.queue = prof_queue;

    status = rocprofiler_open(agent_info->dev_id, feature, feature_count, &context,
            ROCPROFILER_MODE_STANDALONE| ROCPROFILER_MODE_CREATEQUEUE|
            ROCPROFILER_MODE_SINGLEGROUP, &properties);
    TEST_STATUS(status == HSA_STATUS_SUCCESS);

    // Start counters and sample them in the loop with the sampling rate
    status = rocprofiler_start(context, 0);
    TEST_STATUS(status == HSA_STATUS_SUCCESS);

    int numRepeats = 10;
    float alpha = 1.1, beta = 0.9;
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
        read_features(0, context, feature, feature_count);
     }

    // Stop counters
    status = rocprofiler_stop(context, 0);
    TEST_STATUS(status == HSA_STATUS_SUCCESS);

    hipDeviceSynchronize();

    // Finishing cleanup
    // Deleting profiling context will delete all allocated resources
    status = rocprofiler_close(context); 
    TEST_STATUS(status == HSA_STATUS_SUCCESS);

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
