package llama

import "C"

import (
	"runtime"
	"unsafe"
)

func NewBatch(nTokens, embd, nSeqMax int32) *Batch {
	cBatch := C.llama_batch_init(C.int32_t(nTokens), C.int32_t(embd), C.int32_t(nSeqMax))
	b := &Batch{ptr: &cBatch}
	runtime.SetFinalizer(b, (*Batch).Free)
	return b
}

func (batch *Batch) Add(token Token, pos Pos, seqIDs []SeqID, logits bool) {
	if batch == nil || batch.ptr == nil || seqIDs == nil || len(seqIDs) == 0 {
		return
	}
	// Capacity check: we compare n_tokens against total capacity derived from slices
	// We cannot directly read capacity; rely on user to size appropriately.
	idx := int(batch.ptr.n_tokens)
	C.llama_batch_add_wrapper(batch.ptr, C.int(idx), C.llama_token(token), C.llama_pos(pos), C.llama_seq_id(seqIDs[0]), C.bool(logits))
	batch.ptr.n_tokens++
}

func (batch *Batch) getTokenSlice() []Token {
	if batch.ptr == nil || batch.ptr.token == nil {
		return nil
	}
	return (*[1 << 30]Token)(unsafe.Pointer(batch.ptr.token))[:batch.ptr.n_tokens:batch.ptr.n_tokens]
}

func (batch *Batch) Free() {
	if batch.ptr != nil {
		C.llama_batch_free_seq_ids(batch.ptr)
		C.llama_batch_free(*batch.ptr)
		batch.ptr = nil
	}
}
