#include <iostream>
#include <cstring>
#include<roctracer.h>
#include<roctracer_kfd.h>

const char* kfd_api_id[KFD_API_ID_NUMBER] = {
    "hsaKmtOpenKFD",
    "hsaKmtCloseKFD",
    "hsaKmtGetVersion",
    "hsaKmtAcquireSystemProperties",
    "hsaKmtReleaseSystemProperties",
    "hsaKmtGetNodeProperties",
    "hsaKmtGetNodeMemoryProperties",
    "hsaKmtGetNodeCacheProperties",
    "hsaKmtGetNodeIoLinkProperties",
    "hsaKmtCreateEvent",
    "hsaKmtDestroyEvent",
    "hsaKmtSetEvent",
    "hsaKmtResetEvent",
    "hsaKmtQueryEventState",
    "hsaKmtWaitOnEvent",
    "hsaKmtWaitOnMultipleEvents",
    "hsaKmtReportQueue",
    "hsaKmtCreateQueue",
    "hsaKmtUpdateQueue",
    "hsaKmtDestroyQueue",
    "hsaKmtSetQueueCUMask",
    "hsaKmtGetQueueInfo",
    "hsaKmtSetMemoryPolicy",
    "hsaKmtAllocMemory",
    "hsaKmtFreeMemory",
    "hsaKmtRegisterMemory",
    "hsaKmtRegisterMemoryToNodes",
    "hsaKmtRegisterMemoryWithFlags",
    "hsaKmtRegisterGraphicsHandleToNodes",
    "hsaKmtShareMemory",
    "hsaKmtRegisterSharedHandle",
    "hsaKmtRegisterSharedHandleToNodes",
    "hsaKmtProcessVMRead",
    "hsaKmtProcessVMWrite",
    "hsaKmtDeregisterMemory",
    "hsaKmtMapMemoryToGPU",
    "hsaKmtMapMemoryToGPUNodes",
    "hsaKmtUnmapMemoryToGPU",
    "hsaKmtMapGraphicHandle",
    "hsaKmtUnmapGraphicHandle",
    "hsaKmtAllocQueueGWS",
    "hsaKmtDbgRegister",
    "hsaKmtDbgUnregister",
    "hsaKmtDbgWavefrontControl",
    "hsaKmtDbgAddressWatch",
    "hsaKmtQueueSuspend",
    "hsaKmtQueueResume",
    "hsaKmtEnableDebugTrap",
    "hsaKmtEnableDebugTrapWithPollFd",
    "hsaKmtDisableDebugTrap",
    "hsaKmtQueryDebugEvent",
    "hsaKmtGetQueueSnapshot",
    "hsaKmtSetWaveLaunchTrapOverride",
    "hsaKmtSetWaveLaunchMode",
    "hsaKmtGetKernelDebugTrapVersionInfo",
    "hsaKmtGetThunkDebugTrapVersionInfo",
    "hsaKmtSetAddressWatch",
    "hsaKmtClearAddressWatch",
    "hsaKmtEnablePreciseMemoryOperations",
    "hsaKmtDisablePreciseMemoryOperations",
    "hsaKmtGetClockCounters",
    "hsaKmtPmcGetCounterProperties",
    "hsaKmtPmcRegisterTrace",
    "hsaKmtPmcUnregisterTrace",
    "hsaKmtPmcAcquireTraceAccess",
    "hsaKmtPmcReleaseTraceAccess",
    "hsaKmtPmcStartTrace",
    "hsaKmtPmcQueryTrace",
    "hsaKmtPmcStopTrace",
    "hsaKmtSetTrapHandler",
    "hsaKmtGetTileConfig",
    "hsaKmtQueryPointerInfo",
    "hsaKmtSetMemoryUserData",
    "hsaKmtSPMAcquire",
    "hsaKmtSPMRelease",
    "hsaKmtSPMSetDestBuffer"
};

bool kfd_domain_api_compatibility_check() {
    uint32_t domain, op, kind, res_cnt = 0;
    bool val_res = 0;
    std::cout << "Checking KFD API Domain ID compatibility Start" << std::endl;
    domain = ACTIVITY_DOMAIN_KFD_API;
    kind = 0;
    for (unsigned int i = 0; i < KFD_API_ID_NUMBER; i++) {
        op = i;
        const char* op_string = roctracer_op_string(domain, op, kind);
        printf("%s op_string\n", op_string);
        for (unsigned int j = 0; j < KFD_API_ID_NUMBER; j++) {
            if(!strcmp(op_string, kfd_api_id[j])) {
                val_res = 1;
                break;
            } else {
                val_res = 0;
            }
        }
        if (val_res == 0) {
            res_cnt++;
            std::cout << "KFD API Domain ID compatibility Failed for : ";
            std::cout << op_string << std::endl;
        }
    }
    std::cout << "Checking KFD API Domain ID compatibility Stop" << std::endl;
    if (res_cnt > 0)
        return 0;
    else
        return 1;
}

int main() {
 
    uint32_t roctracer_major_ver = roctracer_version_major();
    std::cout << "ROCTracer Major version " << roctracer_major_ver << std::endl;
    if (kfd_domain_api_compatibility_check()) {
        std::cout << "hip_domain_api_compatibility_check : PASS" << std::endl;
    } else {
        std::cout << "hip_domain_api_compatibility_check : FAIL" << std::endl;
    }

    return 0;
}
