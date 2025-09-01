package llama

/*
#include "llama.h"
*/
import "C"

// Aliases to C types
type Token = C.llama_token
type Pos = C.llama_pos
type SeqID = C.llama_seq_id

// Opaque wrappers
type Model struct{ ptr *C.struct_llama_model }
type Context struct{ ptr *C.struct_llama_context }
type Memory struct{ ptr C.llama_memory_t }
type Vocab struct{ ptr *C.struct_llama_vocab }

type LLama struct {
	Model   *Model
	Context *Context
}

type Batch struct{ ptr *C.struct_llama_batch }
type Sampler struct{ ptr *C.struct_llama_sampler }

// Parameters
type ModelParams struct {
	NGPULayers   int32
	SplitMode    SplitMode
	MainGPU      int32
	TensorSplit  []float32 // not passed to C here; extend if needed
	VocabOnly    bool
	UseMMmap     bool
	UseMLock     bool
	CheckTensors bool
}

type ContextParams struct {
	NCtx            uint32
	NBatch          uint32
	NUBatch         uint32
	NSeqMax         uint32
	NThreads        int32
	NThreadsBatch   int32
	RopeScalingType RopeScalingType
	PoolingType     PoolingType
	AttentionType   AttentionType
	RopeFreqBase    float32
	RopeFreqScale   float32
	FlashAttn       bool
	OffloadKQV      bool
	Embeddings      bool
	NoPerf          bool
	TypeK           GGMLType
	TypeV           GGMLType
}

// Enums
type SplitMode int

const (
	SPLIT_MODE_NONE  SplitMode = C.LLAMA_SPLIT_MODE_NONE
	SPLIT_MODE_LAYER SplitMode = C.LLAMA_SPLIT_MODE_LAYER
	SPLIT_MODE_ROW   SplitMode = C.LLAMA_SPLIT_MODE_ROW
)

type RopeScalingType int

const (
	ROPE_SCALING_TYPE_UNSPECIFIED RopeScalingType = C.LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED
	ROPE_SCALING_TYPE_NONE        RopeScalingType = C.LLAMA_ROPE_SCALING_TYPE_NONE
	ROPE_SCALING_TYPE_LINEAR      RopeScalingType = C.LLAMA_ROPE_SCALING_TYPE_LINEAR
	ROPE_SCALING_TYPE_YARN        RopeScalingType = C.LLAMA_ROPE_SCALING_TYPE_YARN
)

type PoolingType int

const (
	POOLING_TYPE_UNSPECIFIED PoolingType = C.LLAMA_POOLING_TYPE_UNSPECIFIED
	POOLING_TYPE_NONE        PoolingType = C.LLAMA_POOLING_TYPE_NONE
	POOLING_TYPE_MEAN        PoolingType = C.LLAMA_POOLING_TYPE_MEAN
	POOLING_TYPE_CLS         PoolingType = C.LLAMA_POOLING_TYPE_CLS
	POOLING_TYPE_LAST        PoolingType = C.LLAMA_POOLING_TYPE_LAST
)

type AttentionType int

const (
	ATTENTION_TYPE_UNSPECIFIED AttentionType = C.LLAMA_ATTENTION_TYPE_UNSPECIFIED
	ATTENTION_TYPE_CAUSAL      AttentionType = C.LLAMA_ATTENTION_TYPE_CAUSAL
	ATTENTION_TYPE_NON_CAUSAL  AttentionType = C.LLAMA_ATTENTION_TYPE_NON_CAUSAL
)

type GGMLType int

const (
	TYPE_F32  GGMLType = C.GGML_TYPE_F32
	TYPE_F16  GGMLType = C.GGML_TYPE_F16
	TYPE_Q4_0 GGMLType = C.GGML_TYPE_Q4_0
	TYPE_Q4_1 GGMLType = C.GGML_TYPE_Q4_1
	TYPE_Q5_0 GGMLType = C.GGML_TYPE_Q5_0
	TYPE_Q5_1 GGMLType = C.GGML_TYPE_Q5_1
	TYPE_Q8_0 GGMLType = C.GGML_TYPE_Q8_0
)

type ChatMessage struct {
	Role    string
	Content string
}

type PerfContextData struct {
	TStartMs float64
	TLoadMs  float64
	TPEvalMs float64
	TEvalMs  float64
	NPEval   int32
	NEval    int32
	NReused  int32
}

// Defaults
func DefaultModelParams() ModelParams {
	c := C.llama_model_default_params()
	return ModelParams{
		NGPULayers:   c.n_gpu_layers,
		SplitMode:    SplitMode(c.split_mode),
		MainGPU:      c.main_gpu,
		VocabOnly:    bool(c.vocab_only),
		UseMMmap:     bool(c.use_mmap),
		UseMLock:     bool(c.use_mlock),
		CheckTensors: bool(c.check_tensors),
	}
}

func DefaultContextParams() ContextParams {
	c := C.llama_context_default_params()
	return ContextParams{
		NCtx:            c.n_ctx,
		NBatch:          c.n_batch,
		NUBatch:         c.n_ubatch,
		NSeqMax:         c.n_seq_max,
		NThreads:        c.n_threads,
		NThreadsBatch:   c.n_threads_batch,
		RopeScalingType: RopeScalingType(c.rope_scaling_type),
		PoolingType:     PoolingType(c.pooling_type),
		AttentionType:   AttentionType(c.attention_type),
		RopeFreqBase:    float32(c.rope_freq_base),
		RopeFreqScale:   float32(c.rope_freq_scale),
		FlashAttn:       bool(c.flash_attn),
		OffloadKQV:      bool(c.offload_kqv),
		Embeddings:      bool(c.embeddings),
		NoPerf:          bool(c.no_perf),
		TypeK:           GGMLType(c.type_k),
		TypeV:           GGMLType(c.type_v),
	}
}

func DefaultSamplerChainParams() C.struct_llama_sampler_chain_params {
	return C.llama_sampler_chain_default_params()
}
