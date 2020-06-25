#include <iostream>
#include <cstring>
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include<roctracer.h>
#include<roctracer_hip.h>

const char* hip_api_id[HIP_API_ID_NUMBER] = {
    "hipDrvMemcpy3DAsync",
    "hipDeviceEnablePeerAccess",
    "hipStreamCreateWithPriority",
    "hipMemcpyToSymbolAsync",
    "hipModuleUnload",
    "hipMallocPitch",
    "hipMalloc",
    "hipMemsetD16",
    "hipDeviceGetName",
    "hipEventRecord",
    "hipCtxSynchronize",
    "hipSetDevice",
    "hipCtxGetApiVersion",
    "hipMemcpyFromSymbolAsync",
    "hipExtGetLinkTypeAndHopCount",
    "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor",
    "hipMemset3D",
    "hipHostFree",
    "hipMemcpy2DToArray",
    "hipMemsetD8Async",
    "hipCtxGetCacheConfig",
    "hipStreamWaitEvent",
    "hipDeviceGetStreamPriorityRange",
    "hipModuleLoad",
    "hipDevicePrimaryCtxSetFlags",
    "hipLaunchCooperativeKernel",
    "hipLaunchCooperativeKernelMultiDevice",
    "hipMemcpyAsync",
    "hipMalloc3DArray",
    "hipMallocHost",
    "hipCtxGetCurrent",
    "hipDevicePrimaryCtxGetState",
    "hipEventQuery",
    "hipEventCreate",
    "hipMemGetAddressRange",
    "hipMemcpyFromSymbol",
    "hipArrayCreate",
    "hipStreamGetFlags",
    "hipMallocArray",
    "hipCtxGetSharedMemConfig",
    "hipModuleOccupancyMaxPotentialBlockSize",
    "hipMemPtrGetInfo",
    "hipFuncGetAttribute",
    "hipCtxGetFlags",
    "hipStreamDestroy",
    "__hipPushCallConfiguration",
    "hipMemset3DAsync",
    "hipMemcpy3D",
    "hipInit",
    "hipMemcpyAtoH",
    "hipStreamGetPriority",
    "hipMemset2D",
    "hipMemset2DAsync",
    "hipDeviceCanAccessPeer",
    "hipLaunchByPtr",
    "hipLaunchKernel",
    "hipMemsetD16Async",
    "hipDeviceGetByPCIBusId",
    "hipHostUnregister",
    "hipProfilerStop",
    "hipExtStreamCreateWithCUMask",
    "hipStreamSynchronize",
    "hipFreeHost",
    "hipDeviceSetCacheConfig",
    "hipGetErrorName",
    "hipMemcpyHtoD",
    "hipModuleGetGlobal",
    "hipMemcpyHtoA",
    "hipCtxCreate",
    "hipMemcpy2D",
    "hipIpcCloseMemHandle",
    "hipChooseDevice",
    "hipDeviceSetSharedMemConfig",
    "hipDeviceComputeCapability",
    "hipMallocMipmappedArray",
    "hipSetupArgument",
    "hipProfilerStart",
    "hipCtxSetCacheConfig",
    "hipFuncSetCacheConfig",
    "hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
    "hipModuleGetTexRef",
    "hipMemcpyPeerAsync",
    "hipMemcpyWithStream",
    "hipConfigureCall",
    "hipGetMipmappedArrayLevel",
    "hipMemcpy3DAsync",
    "hipEventDestroy",
    "hipCtxPopCurrent",
    "hipGetSymbolAddress",
    "hipHostGetFlags",
    "hipHostMalloc",
    "hipDriverGetVersion",
    "hipFreeMipmappedArray",
    "hipMemGetInfo",
    "hipDeviceReset",
    "hipMemset",
    "hipMemsetD8",
    "hipMemcpyParam2DAsync",
    "hipHostRegister",
    "hipCtxSetSharedMemConfig",
    "hipArray3DCreate",
    "hipIpcOpenMemHandle",
    "hipGetLastError",
    "hipGetDeviceFlags",
    "hipDeviceGetSharedMemConfig",
    "hipDrvMemcpy3D",
    "hipMemcpy2DFromArray",
    "hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
    "hipSetDeviceFlags",
    "hipHccModuleLaunchKernel",
    "hipFree",
    "hipOccupancyMaxPotentialBlockSize",
    "hipDeviceGetAttribute",
    "hipMemcpyDtoH",
    "hipCtxDisablePeerAccess",
    "hipMallocManaged",
    "hipCtxDestroy",
    "hipIpcGetMemHandle",
    "hipMemcpyHtoDAsync",
    "hipCtxGetDevice",
    "hipMemcpyDtoD",
    "hipModuleLoadData",
    "hipDeviceTotalMem",
    "hipOccupancyMaxActiveBlocksPerMultiprocessor",
    "hipCtxSetCurrent",
    "hipGetErrorString",
    "hipStreamCreate",
    "hipDevicePrimaryCtxRetain",
    "hipDeviceDisablePeerAccess",
    "hipStreamCreateWithFlags",
    "hipMemcpyFromArray",
    "hipMemcpy2DAsync",
    "hipFuncGetAttributes",
    "hipGetSymbolSize",
    "hipIpcGetEventHandle",
    "hipEventCreateWithFlags",
    "hipStreamQuery",
    "hipDeviceGetPCIBusId",
    "hipMemcpy",
    "hipPeekAtLastError",
    "hipExtLaunchMultiKernelMultiDevice",
    "hipStreamAddCallback",
    "hipMemcpyToArray",
    "hipMemsetD32",
    "hipExtModuleLaunchKernel",
    "hipDeviceSynchronize",
    "hipDeviceGetCacheConfig",
    "hipMalloc3D",
    "hipPointerGetAttributes",
    "hipMemsetAsync",
    "hipMemcpyToSymbol",
    "hipModuleOccupancyMaxPotentialBlockSizeWithFlags",
    "hipCtxPushCurrent",
    "hipMemcpyPeer",
    "hipEventSynchronize",
    "hipMemcpyDtoDAsync",
    "hipExtMallocWithFlags",
    "hipCtxEnablePeerAccess",
    "hipMemAllocHost",
    "hipMemcpyDtoHAsync",
    "hipModuleLaunchKernel",
    "hipMemAllocPitch",
    "hipExtLaunchKernel",
    "hipMemcpy2DFromArrayAsync",
    "hipDeviceGetLimit",
    "hipModuleLoadDataEx",
    "hipRuntimeGetVersion",
    "__hipPopCallConfiguration",
    "hipGetDeviceProperties",
    "hipDeviceGet",
    "hipFreeArray",
    "hipEventElapsedTime",
    "hipDevicePrimaryCtxRelease",
    "hipHostGetDevicePointer",
    "hipMemcpyParam2D",
    "hipDevicePrimaryCtxReset",
    "hipModuleGetFunction",
    "hipMemsetD32Async",
    "hipGetDevice",
    "hipGetDeviceCount",
    "hipIpcOpenEventHandle"
};

bool hip_domain_api_compatibility_check(uint32_t rt_major_ver) {
    uint32_t domain, op, kind, res_cnt = 0;
    bool val_res = 0;
    std::cout << "ROCTracer Major version " << rt_major_ver << std::endl;
    std::cout << "Checking HIP API Domain ID compatibility Start" << std::endl;
    domain = ACTIVITY_DOMAIN_HIP_API;
    kind = 0;
    for (unsigned int i = 0; i < HIP_API_ID_NUMBER; i++) {
        op = i;
        const char* op_string = roctracer_op_string(domain, op, kind);
        printf("%s op_string\n", op_string);
        for (unsigned int j = 0; j < HIP_API_ID_NUMBER; j++) {
            if(!strcmp(op_string, hip_api_id[j])) {
                val_res = 1;
                break;
            } else {
                val_res = 0;
            }
        }
        if (val_res == 0) {
            res_cnt++;
            std::cout << "HIP API Domain ID compatibility Failed for : ";
            std::cout << op_string << std::endl;
        }
    }
    std::cout << "Checking HIP API Domain ID compatibility Stop" << std::endl;
    if (res_cnt > 0)
        return 0;
    else
        return 1;
}

int main() {
 
    uint32_t roctracer_major_ver = roctracer_version_major();
    if (hip_domain_api_compatibility_check(roctracer_major_ver)) {
        std::cout << "hip_domain_api_compatibility_check : PASS" << std::endl;
        return 0;
    } else {
        std::cout << "hip_domain_api_compatibility_check : FAIL" << std::endl;
        return -1;
    }
}
