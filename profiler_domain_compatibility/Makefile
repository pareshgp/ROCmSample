HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc -std=c++11 -DAMD_INTERNAL_BUILD
HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --compiler)
INC=-I/opt/rocm/roctracer/include/ -I/opt/rocm/hsa/include/hsa/ -I/opt/rocm/include/ -I/opt/rocm/rocblas/include
LIB=-L/opt/rocm/roctracer/lib -l roctracer64 -l roctx64 -L/opt/rocm/rocblas/lib -l rocblas 

all: roctracer_domain_hip_api_val roctracer_domain_hsa_api_val roctracer_domain_kfd_api_val

roctracer_domain_hip_api_val: roctracer_domain_hip_api_val.cpp
	$(HIPCC) $(HIPCC_FLAGS) $(INC) $(LIB) -g $< -o $@

roctracer_domain_hsa_api_val: roctracer_domain_hsa_api_val.cpp
	$(HIPCC) $(HIPCC_FLAGS) $(INC) $(LIB) -g $< -o $@

roctracer_domain_kfd_api_val: roctracer_domain_kfd_api_val.cpp
	$(HIPCC) $(HIPCC_FLAGS) $(INC) $(LIB) -g $< -o $@

clean:
	rm -f  *.o
