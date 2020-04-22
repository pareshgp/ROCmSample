/******************************************************************************
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*******************************************************************************/

#include <hsa.h>
#include <string.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <atomic>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <rocblas.h>
#include <string.h>
#include <unistd.h>
#include <iostream>
#include <rocprofiler.h>
#include "hsa_rsrc_factory.h"

// Dispatch callbacks and context handlers synchronization
pthread_mutex_t mutex = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;

// Error handler
void fatal(const std::string msg) {
  fflush(stdout);
  fprintf(stderr, "%s\n\n", msg.c_str());
  fflush(stderr);
  abort();
}

// Check returned HSA API status
void check_status(hsa_status_t status) {
  if (status != HSA_STATUS_SUCCESS) {
    const char* error_string = NULL;
    rocprofiler_error_string(&error_string);
    fprintf(stderr, "ERROR: %s\n", error_string);
    abort();
  }
}

// Context stored entry type
struct context_entry_t {
  bool valid;
  hsa_agent_t agent;
  rocprofiler_group_t group;
  rocprofiler_callback_data_t data;
};

// Dump stored context entry
void dump_context_entry(context_entry_t* entry) {
  volatile std::atomic<bool>* valid = reinterpret_cast<std::atomic<bool>*>(&entry->valid);
  while (valid->load() == false) sched_yield();

  const std::string kernel_name = entry->data.kernel_name;
  const rocprofiler_dispatch_record_t* record = entry->data.record;

  fflush(stdout);
  fprintf(stdout, "kernel-object(0x%lx) name(\"%s\")", entry->data.kernel_object, kernel_name.c_str());
  if (record) fprintf(stdout, ", gpu-id(%u), time(%lu,%lu,%lu,%lu)",
    HsaRsrcFactory::Instance().GetAgentInfo(entry->agent)->dev_index,
    record->dispatch,
    record->begin,
    record->end,
    record->complete);
  fprintf(stdout, "\n");
  fflush(stdout);

  rocprofiler_group_t& group = entry->group;
  if (group.context == NULL) {
    fprintf(stderr, "tool error: context is NULL\n");
    abort();
  }

  rocprofiler_close(group.context);
}

// Profiling completion handler
// Dump and delete the context entry
// Return true if the context was dumped successfully
bool context_handler(rocprofiler_group_t group, void* arg) {
  context_entry_t* entry = reinterpret_cast<context_entry_t*>(arg);

  if (pthread_mutex_lock(&mutex) != 0) {
    perror("pthread_mutex_lock");
    abort();
  }

  dump_context_entry(entry);
  delete entry;

  if (pthread_mutex_unlock(&mutex) != 0) {
    perror("pthread_mutex_unlock");
    abort();
  }

  return false;
}

// Kernel disoatch callback
hsa_status_t dispatch_callback(const rocprofiler_callback_data_t* callback_data, void* /*user_data*/,
                               rocprofiler_group_t* group) {
  // HSA status
  hsa_status_t status = HSA_STATUS_ERROR;

  // Profiling context
  rocprofiler_t* context = NULL;

  // Context entry
  context_entry_t* entry = new context_entry_t();

  // context properties
  rocprofiler_properties_t properties{};
  properties.handler = context_handler;
  properties.handler_arg = (void*)entry;

  // Open profiling context
  status = rocprofiler_open(callback_data->agent, NULL, 0,
                            &context, 0 /*ROCPROFILER_MODE_SINGLEGROUP*/, &properties);
  check_status(status);

  // Get group[0]
  status = rocprofiler_get_group(context, 0, group);
  check_status(status);

  // Fill profiling context entry
  entry->agent = callback_data->agent;
  entry->group = *group;
  entry->data = *callback_data;
  entry->data.kernel_name = strdup(callback_data->kernel_name);
  reinterpret_cast<std::atomic<bool>*>(&entry->valid)->store(true);

  return HSA_STATUS_SUCCESS;
}
rocblas_operation transa = rocblas_operation_none;
rocblas_operation transb = rocblas_operation_transpose;

float fast_pseudo_rand(u_long *nextr) {
    *nextr = *nextr * 1103515245 + 12345;
    return static_cast<float>(static_cast<uint32_t>
                    ((*nextr / 65536) % 320000)) / 0.1234;
}

int run_rocblas_sgemm(){
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


    // copy from Host to device start
    if (da) {
        if (hipMemcpy(da, ha, sizeof(float) * size_a, hipMemcpyHostToDevice)
            != hipSuccess) {
            std::cout << "hipMemcpyHostToDevice failed" ;
            return -1;
        }
        std::cout << "hipMemcpyHostToDevice done 1" ;
    }

    if (db) {
        if (hipMemcpy(db, hb, sizeof(float) * size_b, hipMemcpyHostToDevice)
            != hipSuccess) {
            std::cout << "hipMemcpyHostToDevice failed" ;
            return -1;
        }
        std::cout << "hipMemcpyHostToDevice done 2" ;
    }

    if (dc) {
        if (hipMemcpy(dc, hc, sizeof(float) * size_c, hipMemcpyHostToDevice)
            != hipSuccess) {
            std::cout << "hipMemcpyHostToDevice failed" ;
            return -1;
        }
        std::cout << "hipMemcpyHostToDevice done 3" ;
    }

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
     }

    hipDeviceSynchronize();

    // releases the host matrix memory
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

int main() {
  int ret_val;
  const unsigned kiter = 25;

  // Adding dispatch observer
  rocprofiler_queue_callbacks_t callbacks_ptrs{};
  callbacks_ptrs.dispatch = dispatch_callback;
  rocprofiler_set_queue_callbacks(callbacks_ptrs, NULL);

  // Instantiate HSA resources
  HsaRsrcFactory::Create();

  // Getting GPU device info
  const AgentInfo* agent_info = NULL;
  if (HsaRsrcFactory::Instance().GetGpuAgentInfo(0, &agent_info) == false) abort();

  // Creating the queue
  hsa_queue_t* queue = NULL;
  if (HsaRsrcFactory::Instance().CreateQueue(agent_info, 128, &queue) == false) abort();


  for (unsigned ind = 0; ind < kiter; ++ind) {
    printf("Iteration %u:\n", ind);
    if ((ind & 1) == 0) rocprofiler_start_queue_callbacks();
    else rocprofiler_stop_queue_callbacks();
    ret_val = run_rocblas_sgemm(); 
  }


  return (ret_val == 0) ? 0 : 1;
}
