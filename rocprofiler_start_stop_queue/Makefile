HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --compiler)
INC=-I/opt/rocm/rocblas/include -I/opt/rocm/rocprofiler/include/ -I/opt/rocm/hsa/include/hsa
LIB=-L/opt/rocm/rocblas/lib -l rocblas -L/opt/rocm/rocprofiler/lib -l rocprofiler64

#all: rocprofiler_start_stop

#rocprofiler_start_stop: rocprofiler_start_stop.cpp
#	$(HIPCC) $(HIPCC_FLAGS) $(INC) $(LIB) -g $< -o $@

HIPCC=$(HIP_PATH)/bin/hipcc -DAMD_INTERNAL_BUILD -ldl

#TARGET=hcc

SOURCES = hsa_rsrc_factory.cpp stand_intercept_test.cpp
OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE=./rocprofiler_start_stop

all: clean $(EXECUTABLE)

CXXFLAGS =-g $(INC)
CXX=$(HIPCC)

$(EXECUTABLE): $(OBJECTS)
	$(HIPCC) $(OBJECTS) -o $@ $(LIB)


clean:
	rm -f  *.o
