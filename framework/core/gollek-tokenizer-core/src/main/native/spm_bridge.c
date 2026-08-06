#include <sentencepiece_processor.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  sentencepiece::SentencePieceProcessor *processor;
} SpmHandle;

extern "C" {

SpmHandle *spm_create() {
  SpmHandle *handle = (SpmHandle *)malloc(sizeof(SpmHandle));
  handle->processor = new sentencepiece::SentencePieceProcessor();
  return handle;
}

int spm_load(SpmHandle *handle, const char *path) {
  return handle->processor->Load(path).ok() ? 0 : -1;
}

int spm_encode(SpmHandle *handle, const char *input, int **out_ids,
               int *out_len) {
  std::vector<int> ids;
  handle->processor->Encode(input, &ids);

  *out_len = ids.size();
  *out_ids = (int *)malloc(sizeof(int) * (*out_len));
  memcpy(*out_ids, ids.data(), sizeof(int) * (*out_len));
  return 0;
}

int spm_encode_into(SpmHandle *handle, const char *input, int *out_ids,
                    int capacity, int *out_len) {
  std::vector<int> ids;
  handle->processor->Encode(input, &ids);

  int n = (int)ids.size();
  if (n > capacity) {
    n = capacity;
  }
  memcpy(out_ids, ids.data(), sizeof(int) * n);
  *out_len = n;
  return (n == (int)ids.size()) ? 0 : -1;
}

char *spm_decode(SpmHandle *handle, int *ids, int len) {
  std::string output;
  std::vector<int> vec(ids, ids + len);
  handle->processor->Decode(vec, &output);

  char *result = (char *)malloc(output.size() + 1);
  strcpy(result, output.c_str());
  return result;
}

void spm_free_ids(int *ptr) { free(ptr); }

void spm_free_str(char *ptr) { free(ptr); }

void spm_free_string(char *ptr) { free(ptr); }

void spm_destroy(SpmHandle *handle) {
  delete handle->processor;
  free(handle);
}
}
