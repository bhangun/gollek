#ifndef __GOLLEK_SDK_LOCAL_H
#define __GOLLEK_SDK_LOCAL_H

#include <graal_isolate.h>


#if defined(__cplusplus)
extern "C" {
#endif

char* golek_version(graal_isolatethread_t* thread);

long long int golek_client_create(graal_isolatethread_t* thread);

int golek_client_destroy(graal_isolatethread_t* thread, long long int clientHandle);

void golek_client_shutdown_runtime(graal_isolatethread_t* thread);

char* golek_last_error(graal_isolatethread_t* thread);

void golek_clear_last_error(graal_isolatethread_t* thread);

void golek_string_free(graal_isolatethread_t* thread, char* pointer);

char* golek_create_completion_json(graal_isolatethread_t* thread, long long int clientHandle, char* requestJson);

char* golek_create_completion_async_json(graal_isolatethread_t* thread, long long int clientHandle, char* requestJson);

char* golek_stream_completion_json(graal_isolatethread_t* thread, long long int clientHandle, char* requestJson);

char* golek_submit_async_job_json(graal_isolatethread_t* thread, long long int clientHandle, char* requestJson);

char* golek_get_job_status_json(graal_isolatethread_t* thread, long long int clientHandle, char* jobId);

char* golek_wait_for_job_json(graal_isolatethread_t* thread, long long int clientHandle, char* jobId, long long int maxWaitMillis, long long int pollIntervalMillis);

char* golek_batch_inference_json(graal_isolatethread_t* thread, long long int clientHandle, char* batchRequestJson);

char* golek_list_available_providers_json(graal_isolatethread_t* thread, long long int clientHandle);

char* golek_get_provider_info_json(graal_isolatethread_t* thread, long long int clientHandle, char* providerId);

int golek_set_preferred_provider(graal_isolatethread_t* thread, long long int clientHandle, char* providerId);

char* golek_get_preferred_provider_json(graal_isolatethread_t* thread, long long int clientHandle);

char* golek_list_models_json(graal_isolatethread_t* thread, long long int clientHandle);

char* golek_list_models_paginated_json(graal_isolatethread_t* thread, long long int clientHandle, int offset, int limit);

char* golek_get_model_info_json(graal_isolatethread_t* thread, long long int clientHandle, char* modelId);

char* golek_pull_model_json(graal_isolatethread_t* thread, long long int clientHandle, char* modelSpec);

int golek_delete_model(graal_isolatethread_t* thread, long long int clientHandle, char* modelId);

int run_main(int argc, char** argv);

#if defined(__cplusplus)
}
#endif
#endif
