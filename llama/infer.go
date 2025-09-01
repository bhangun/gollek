package llama

/*
#include "llama.h"
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// Encode/Decode + logits helpers
func (llm *LLama) Encode(batch *Batch) error {
	if batch == nil || batch.ptr == nil {
		return nil
	}
	r := C.llama_encode(llm.Context.ptr, *batch.ptr)
	if r < 0 {
		return fmtError("encode failed", int(r))
	}
	return nil
}

func (llm *LLama) Decode(batch *Batch) error {
	if batch == nil || batch.ptr == nil {
		return nil
	}
	r := C.llama_decode(llm.Context.ptr, *batch.ptr)
	if r < 0 {
		return fmtError("decode failed", int(r))
	}
	if r == 1 {
		return fmtError("could not find KV slot for batch", 1)
	}
	if r == 2 {
		return fmtError("decode was aborted", 2)
	}
	return nil
}

func (llm *LLama) GetLogits() []float32 {
	cLogits := C.llama_get_logits(llm.Context.ptr)
	if cLogits == nil {
		return nil
	}
	nVocab := C.llama_vocab_n_tokens(C.llama_model_get_vocab(llm.Model.ptr))
	return (*[1 << 30]float32)(unsafe.Pointer(cLogits))[:nVocab:nVocab]
}

func (llm *LLama) GetLogitsIth(i int32) []float32 {
	cLogits := C.llama_get_logits_ith(llm.Context.ptr, C.int32_t(i))
	if cLogits == nil {
		return nil
	}
	nVocab := C.llama_vocab_n_tokens(C.llama_model_get_vocab(llm.Model.ptr))
	return (*[1 << 30]float32)(unsafe.Pointer(cLogits))[:nVocab:nVocab]
}

// small internal helper
func fmtError(msg string, code int) error { return fmt.Errorf("%s with code: %d", msg, code) }
