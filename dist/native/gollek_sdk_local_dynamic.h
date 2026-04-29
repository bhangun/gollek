#ifndef __GOLLEK_SDK_LOCAL_H
#define __GOLLEK_SDK_LOCAL_H

#include <graal_isolate_dynamic.h>


#if defined(__cplusplus)
extern "C" {
#endif

typedef char* (*golek_version_fn_t)(graal_isolatethread_t* thread);

typedef long long int (*golek_client_create_fn_t)(graal_isolatethread_t* thread);

typedef int (*golek_client_destroy_fn_t)(graal_isolatethread_t* thread, long long int clientHandle);

typedef void (*golek_client_shutdown_runtime_fn_t)(graal_isolatethread_t* thread);

typedef char* (*golek_last_error_fn_t)(graal_isolatethread_t* thread);

typedef void (*golek_clear_last_error_fn_t)(graal_isolatethread_t* thread);

typedef void (*golek_string_free_fn_t)(graal_isolatethread_t* thread, char* pointer);

typedef char* (*golek_create_completion_json_fn_t)(graal_isolatethread_t* thread, long long int clientHandle, char* requestJson);

typedef char* (*golek_create_completion_async_json_fn_t)(graal_isolatethread_t* thread, long long int clientHandle, char* requestJson);

typedef char* (*golek_stream_completion_json_fn_t)(graal_isolatethread_t* thread, long long int clientHandle, char* requestJson);

typedef char* (*golek_submit_async_job_json_fn_t)(graal_isolatethread_t* thread, long long int clientHandle, char* requestJson);

typedef char* (*golek_get_job_status_json_fn_t)(graal_isolatethread_t* thread, long long int clientHandle, char* jobId);

typedef char* (*golek_wait_for_job_json_fn_t)(graal_isolatethread_t* thread, long long int clientHandle, char* jobId, long long int maxWaitMillis, long long int pollIntervalMillis);

typedef char* (*golek_batch_inference_json_fn_t)(graal_isolatethread_t* thread, long long int clientHandle, char* batchRequestJson);

typedef char* (*golek_list_available_providers_json_fn_t)(graal_isolatethread_t* thread, long long int clientHandle);

typedef char* (*golek_get_provider_info_json_fn_t)(graal_isolatethread_t* thread, long long int clientHandle, char* providerId);

typedef int (*golek_set_preferred_provider_fn_t)(graal_isolatethread_t* thread, long long int clientHandle, char* providerId);

typedef char* (*golek_get_preferred_provider_json_fn_t)(graal_isolatethread_t* thread, long long int clientHandle);

typedef char* (*golek_list_models_json_fn_t)(graal_isolatethread_t* thread, long long int clientHandle);

typedef char* (*golek_list_models_paginated_json_fn_t)(graal_isolatethread_t* thread, long long int clientHandle, int offset, int limit);

typedef char* (*golek_get_model_info_json_fn_t)(graal_isolatethread_t* thread, long long int clientHandle, char* modelId);

typedef char* (*golek_pull_model_json_fn_t)(graal_isolatethread_t* thread, long long int clientHandle, char* modelSpec);

typedef int (*golek_delete_model_fn_t)(graal_isolatethread_t* thread, long long int clientHandle, char* modelId);

typedef int (*run_main_fn_t)(int argc, char** argv);

#if defined(__cplusplus)
}
#endif
#endif
