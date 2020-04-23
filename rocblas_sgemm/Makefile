HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc -std=c++11 
HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --compiler)
INC=-I/opt/rocm/rocblas/include -I/opt/rocm/roctracer/include/
LIB=-L/opt/rocm/rocblas/lib -l rocblas -L/opt/rocm/roctracer/lib -l roctracer64 -l roctx64

all: rocblas_sgemm perf_overhead

rocblas_sgemm: rocblas_sgemm.cpp
	$(HIPCC) $(HIPCC_FLAGS) $(INC) $(LIB) -g $< -o $@

perf_overhead: small_kernel_perf_over.cpp
	$(HIPCC) $(HIPCC_FLAGS) $(INC) $(LIB) -g $< -o $@
clean:
	rm -f  *.out
