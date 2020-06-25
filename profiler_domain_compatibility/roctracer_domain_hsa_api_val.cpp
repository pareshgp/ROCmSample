#include <iostream>
#include <cstring>
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include<roctracer.h>
#include<roctracer_hsa.h>

const char* hsa_api_id[HSA_API_ID_NUMBER] = {
    "hsa_init",
    "hsa_shut_down",
    "hsa_system_get_info",
    "hsa_system_extension_supported",
    "hsa_system_get_extension_table",
    "hsa_iterate_agents",
    "hsa_agent_get_info",
    "hsa_queue_create",
    "hsa_soft_queue_create",
    "hsa_queue_destroy",
    "hsa_queue_inactivate",
    "hsa_queue_load_read_index_scacquire",
    "hsa_queue_load_read_index_relaxed",
    "hsa_queue_load_write_index_scacquire",
    "hsa_queue_load_write_index_relaxed",
    "hsa_queue_store_write_index_relaxed",
    "hsa_queue_store_write_index_screlease",
    "hsa_queue_cas_write_index_scacq_screl",
    "hsa_queue_cas_write_index_scacquire",
    "hsa_queue_cas_write_index_relaxed",
    "hsa_queue_cas_write_index_screlease",
    "hsa_queue_add_write_index_scacq_screl",
    "hsa_queue_add_write_index_scacquire",
    "hsa_queue_add_write_index_relaxed",
    "hsa_queue_add_write_index_screlease",
    "hsa_queue_store_read_index_relaxed",
    "hsa_queue_store_read_index_screlease",
    "hsa_agent_iterate_regions",
    "hsa_region_get_info",
    "hsa_agent_get_exception_policies",
    "hsa_agent_extension_supported",
    "hsa_memory_register",
    "hsa_memory_deregister",
    "hsa_memory_allocate",
    "hsa_memory_free",
    "hsa_memory_copy",
    "hsa_memory_assign_agent",
    "hsa_signal_create",
    "hsa_signal_destroy",
    "hsa_signal_load_relaxed",
    "hsa_signal_load_scacquire",
    "hsa_signal_store_relaxed",
    "hsa_signal_store_screlease",
    "hsa_signal_wait_relaxed",
    "hsa_signal_wait_scacquire",
    "hsa_signal_and_relaxed",
    "hsa_signal_and_scacquire",
    "hsa_signal_and_screlease",
    "hsa_signal_and_scacq_screl",
    "hsa_signal_or_relaxed",
    "hsa_signal_or_scacquire",
    "hsa_signal_or_screlease",
    "hsa_signal_or_scacq_screl",
    "hsa_signal_xor_relaxed",
    "hsa_signal_xor_scacquire",
    "hsa_signal_xor_screlease",
    "hsa_signal_xor_scacq_screl",
    "hsa_signal_exchange_relaxed",
    "hsa_signal_exchange_scacquire",
    "hsa_signal_exchange_screlease",
    "hsa_signal_exchange_scacq_screl",
    "hsa_signal_add_relaxed",
    "hsa_signal_add_scacquire",
    "hsa_signal_add_screlease",
    "hsa_signal_add_scacq_screl",
    "hsa_signal_subtract_relaxed",
    "hsa_signal_subtract_scacquire",
    "hsa_signal_subtract_screlease",
    "hsa_signal_subtract_scacq_screl",
    "hsa_signal_cas_relaxed",
    "hsa_signal_cas_scacquire",
    "hsa_signal_cas_screlease",
    "hsa_signal_cas_scacq_screl",
    "hsa_isa_from_name",
    "hsa_isa_get_info",
    "hsa_isa_compatible",
    "hsa_code_object_serialize",
    "hsa_code_object_deserialize",
    "hsa_code_object_destroy",
    "hsa_code_object_get_info",
    "hsa_code_object_get_symbol",
    "hsa_code_symbol_get_info",
    "hsa_code_object_iterate_symbols",
    "hsa_executable_create",
    "hsa_executable_destroy",
    "hsa_executable_load_code_object",
    "hsa_executable_freeze",
    "hsa_executable_get_info",
    "hsa_executable_global_variable_define",
    "hsa_executable_agent_global_variable_define",
    "hsa_executable_readonly_variable_define",
    "hsa_executable_validate",
    "hsa_executable_get_symbol",
    "hsa_executable_symbol_get_info",
    "hsa_executable_iterate_symbols",
    "hsa_status_string",
    "hsa_extension_get_name",
    "hsa_system_major_extension_supported",
    "hsa_system_get_major_extension_table",
    "hsa_agent_major_extension_supported",
    "hsa_cache_get_info",
    "hsa_agent_iterate_caches",
    "hsa_signal_silent_store_relaxed",
    "hsa_signal_silent_store_screlease",
    "hsa_signal_group_create",
    "hsa_signal_group_destroy",
    "hsa_signal_group_wait_any_scacquire",
    "hsa_signal_group_wait_any_relaxed",
    "hsa_agent_iterate_isas",
    "hsa_isa_get_info_alt",
    "hsa_isa_get_exception_policies",
    "hsa_isa_get_round_method",
    "hsa_wavefront_get_info",
    "hsa_isa_iterate_wavefronts",
    "hsa_code_object_get_symbol_from_name",
    "hsa_code_object_reader_create_from_file",
    "hsa_code_object_reader_create_from_memory",
    "hsa_code_object_reader_destroy",
    "hsa_executable_create_alt",
    "hsa_executable_load_program_code_object",
    "hsa_executable_load_agent_code_object",
    "hsa_executable_validate_alt",
    "hsa_executable_get_symbol_by_name",
    "hsa_executable_iterate_agent_symbols",
    "hsa_executable_iterate_program_symbols",
    "hsa_amd_coherency_get_type",
    "hsa_amd_coherency_set_type",
    "hsa_amd_profiling_set_profiler_enabled",
    "hsa_amd_profiling_async_copy_enable",
    "hsa_amd_profiling_get_dispatch_time",
    "hsa_amd_profiling_get_async_copy_time",
    "hsa_amd_profiling_convert_tick_to_system_domain",
    "hsa_amd_signal_async_handler",
    "hsa_amd_async_function",
    "hsa_amd_signal_wait_any",
    "hsa_amd_queue_cu_set_mask",
    "hsa_amd_memory_pool_get_info",
    "hsa_amd_agent_iterate_memory_pools",
    "hsa_amd_memory_pool_allocate",
    "hsa_amd_memory_pool_free",
    "hsa_amd_memory_async_copy",
    "hsa_amd_agent_memory_pool_get_info",
    "hsa_amd_agents_allow_access",
    "hsa_amd_memory_pool_can_migrate",
    "hsa_amd_memory_migrate",
    "hsa_amd_memory_lock",
    "hsa_amd_memory_unlock",
    "hsa_amd_memory_fill",
    "hsa_amd_interop_map_buffer",
    "hsa_amd_interop_unmap_buffer",
    "hsa_amd_image_create",
    "hsa_amd_pointer_info",
    "hsa_amd_pointer_info_set_userdata",
    "hsa_amd_ipc_memory_create",
    "hsa_amd_ipc_memory_attach",
    "hsa_amd_ipc_memory_detach",
    "hsa_amd_signal_create",
    "hsa_amd_ipc_signal_create",
    "hsa_amd_ipc_signal_attach",
    "hsa_amd_register_system_event_handler",
    "hsa_amd_queue_intercept_create",
    "hsa_amd_queue_intercept_register",
    "hsa_amd_queue_set_priority",
    "hsa_amd_memory_async_copy_rect",
    "hsa_amd_runtime_queue_create_register",
    "hsa_amd_memory_lock_to_pool",
    "hsa_amd_register_deallocation_callback",
    "hsa_amd_deregister_deallocation_callback",
    "hsa_ext_image_get_capability",
    "hsa_ext_image_data_get_info",
    "hsa_ext_image_create",
    "hsa_ext_image_import",
    "hsa_ext_image_export",
    "hsa_ext_image_copy",
    "hsa_ext_image_clear",
    "hsa_ext_image_destroy",
    "hsa_ext_sampler_create",
    "hsa_ext_sampler_destroy",
    "hsa_ext_image_get_capability_with_layout",
    "hsa_ext_image_data_get_info_with_layout",
    "hsa_ext_image_create_with_layout",
    "DISPATCH"
};

bool hsa_domain_api_compatibility_check(uint32_t rt_major_ver) {
    uint32_t domain, op, kind, res_cnt = 0;
    bool val_res = 0;
    std::cout << "ROCTracer Major version " << rt_major_ver << std::endl;
    std::cout << "Checking HSA API Domain IP compatibility Start" << std::endl;
    domain = ACTIVITY_DOMAIN_HSA_API;
    kind = 0;
    for (unsigned int i = 0; i < HSA_API_ID_NUMBER; i++) {
        op = i;
        const char* op_string = roctracer_op_string(domain, op, kind);
        printf("%s op_string\n", op_string);
        for (unsigned int j = 0; j < HSA_API_ID_NUMBER; j++) {
            if(!strcmp(op_string, hsa_api_id[j])) {
                val_res = 1;
                break;
            } else {
                val_res = 0;
            }
        }
        if (val_res == 0) {
            res_cnt++;
            std::cout << "HSA API Domain ID compatibility Failed for : ";
            std::cout << op_string << std::endl;
        }
    }
    std::cout << "Checking HSA API Domain ID compatibility Stop" << std::endl;
    if (res_cnt > 0)
        return 0;
    else
        return 1;

}

int main() {
    uint32_t roctracer_major_ver = roctracer_version_major();
    if (hsa_domain_api_compatibility_check(roctracer_major_ver)) {
        std::cout << "hip_domain_api_compatibility_check : PASS" << std::endl;
    } else {
        std::cout << "hip_domain_api_compatibility_check : FAIL" << std::endl;
    }

    return 0;
}
