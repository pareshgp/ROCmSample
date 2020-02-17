HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc -std=c++11 
HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --compiler)
INC=-I/opt/rocm/rocblas/include
LIB=-L/opt/rocm/rocblas/lib -l rocblas

all: rocblas_sgemm

rocblas_sgemm: rocblas_sgemm.cpp
	$(HIPCC) $(HIPCC_FLAGS) $(INC) $(LIB) $< -o $@

clean:
	rm -f  *.out
