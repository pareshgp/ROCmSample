HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc -std=c++11 
HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --compiler)
INC=-I$(HIP_PATH)/../rocblas/include -I$(HIP_PATH)/../roctracer/include/ -I$(HIP_PATH)/../include
LIB=-L$(HIP_PATH)/../rocblas/lib -L$(HIP_PATH)/../lib -l rocblas -L$(HIP_PATH)/../roctracer/lib -l roctracer64 -l roctx64

all: rocblas_sgemm perf_overhead rocblas_roctx pc_test memcpy_val

rocblas_sgemm: rocblas_sgemm.cpp
	$(HIPCC) $(HIPCC_FLAGS) $(INC) $(LIB) -g $< -o $@

perf_overhead: small_kernel_perf_over.cpp
	$(HIPCC) $(HIPCC_FLAGS) $(INC) $(LIB) -g $< -o $@

rocblas_roctx: small_kernel_roctx_perf_over.cpp
	$(HIPCC) $(HIPCC_FLAGS) $(INC) $(LIB) -g $< -o $@

pc_test: pc_test.cpp
	$(HIPCC) $(HIPCC_FLAGS) $(INC) $(LIB) -g $< -o $@

memcpy_val: memcpy_val.cpp
	$(HIPCC) $(HIPCC_FLAGS) $(INC) $(LIB) -g $< -o $@
clean:
	rm -f  *.o rocblas_sgemm perf_overhead rocblas_roctx pc_test memcpy_val
