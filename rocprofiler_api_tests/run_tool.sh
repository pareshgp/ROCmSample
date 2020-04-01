# PATH to custom HSA libs
HSA_PATH=/opt/rocm/hsa

# paths to ROC profiler and oher libraries
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/rocprofiler/lib:/opt/rocm/rocprofiler/tool:$HSA_PATH

# ROC profiler library loaded by HSA runtime
export HSA_TOOLS_LIB=librocprofiler64.so.1
# tool library loaded by ROC profiler
if [ -z "$ROCP_TOOL_LIB" ] ; then
  export ROCP_TOOL_LIB=libintercept_test.so
fi
# enable error messages
export HSA_TOOLS_REPORT_LOAD_FAILURE=1
export HSA_VEN_AMD_AQLPROFILE_LOG=1
export ROCPROFILER_LOG=1
# ROC profiler metrics config file
unset ROCP_PROXY_QUEUE
# ROC profiler metrics config file
if [ -z "$ROCP_METRICS" ] ; then
  export ROCP_METRICS=/opt/rocm/rocprofiler/lib/metrics.xml
fi

