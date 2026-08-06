#include <cstring>
#include <sentencepiece_processor.h>
#include <vector>

extern "C" {

struct SpmHandle {
  sentencepiece::SentencePieceProcessor *processor;
};

SpmHandle *spm_create() {
  return new SpmHandle{new sentencepiece::SentencePieceProcessor()};
}

void spm_destroy(SpmHandle *handle) {
  delete handle->processor;
  delete handle;
}

int spm_load(SpmHandle *handle, const char *model_path) {
  return handle->processor->Load(model_path).ok() ? 0 : -1;
}

/**
 * ZERO-ALLOC encode (caller provides buffer)
 */
int spm_encode_into(SpmHandle *handle, const char *input, int *out_ids,
                    int max_len, int *out_len) {

  std::vector<int> ids;
  handle->processor->Encode(input, &ids);

  int len = ids.size();
  if (len > max_len)
    len = max_len;

  memcpy(out_ids, ids.data(), sizeof(int) * len);
  *out_len = len;

  return 0;
}

char *spm_decode(SpmHandle *handle, int *ids, int len) {
  std::string out;
  handle->processor->Decode(ids, len, &out);

  char *result = (char *)malloc(out.size() + 1);
  strcpy(result, out.c_str());
  return result;
}

void spm_free_string(char *ptr) { free(ptr); }
}