package llama

/*
#include "llama.h"
*/
import "C"

func (mem *Memory) Clear(data bool) {
	C.llama_memory_clear(mem.ptr, C.bool(data))
}

func (mem *Memory) SeqRemove(seqID SeqID, p0, p1 Pos) bool {
	return bool(C.llama_memory_seq_rm(mem.ptr, C.llama_seq_id(seqID), C.llama_pos(p0), C.llama_pos(p1)))
}

func (mem *Memory) SeqCopy(srcSeqID, dstSeqID SeqID, p0, p1 Pos) {
	C.llama_memory_seq_cp(mem.ptr, C.llama_seq_id(srcSeqID), C.llama_seq_id(dstSeqID), C.llama_pos(p0), C.llama_pos(p1))
}

func (mem *Memory) SeqKeep(seqID SeqID) {
	C.llama_memory_seq_keep(mem.ptr, C.llama_seq_id(seqID))
}

func (mem *Memory) SeqAdd(seqID SeqID, p0, p1 Pos, delta Pos) {
	C.llama_memory_seq_add(mem.ptr, C.llama_seq_id(seqID), C.llama_pos(p0), C.llama_pos(p1), C.llama_pos(delta))
}

func (mem *Memory) SeqPosMin(seqID SeqID) Pos {
	return Pos(C.llama_memory_seq_pos_min(mem.ptr, C.llama_seq_id(seqID)))
}
func (mem *Memory) SeqPosMax(seqID SeqID) Pos {
	return Pos(C.llama_memory_seq_pos_max(mem.ptr, C.llama_seq_id(seqID)))
}
func (mem *Memory) CanShift() bool { return bool(C.llama_memory_can_shift(mem.ptr)) }
