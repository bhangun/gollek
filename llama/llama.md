package llama

/*
#cgo CFLAGS: -I./llama.cpp
#cgo CXXFLAGS: -I./llama.cpp -std=c++17
#cgo LDFLAGS: -L./llama.cpp -lllama -lggml -lm -pthread
#cgo darwin LDFLAGS: -framework Accelerate
#cgo linux LDFLAGS: -lrt

#include "llama.h"
#include <stdlib.h>
#include <string.h>

// Helper functions for Go integration
static inline void llama_batch_add_wrapper(struct llama_batch* batch, int idx, llama_token token, llama_pos pos, llama_seq_id seq_id, bool logits) {
    batch->token[idx] = token;
    batch->pos[idx] = pos;
    batch->n_seq_id[idx] = 1;
    batch->seq_id[idx] = malloc(sizeof(llama_seq_id));
    batch->seq_id[idx][0] = seq_id;
    batch->logits[idx] = logits ? 1 : 0;
}

static inline void llama_batch_free_seq_ids(struct llama_batch* batch) {
    for (int i = 0; i < batch->n_tokens; i++) {
        if (batch->seq_id[i] != NULL) {
            free(batch->seq_id[i]);
        }
    }
}

// Chat template wrapper
static inline int llama_chat_apply_template_wrapper(
    const struct llama_model* model,
    const char* tmpl,
    struct llama_chat_message* chat,
    size_t n_msg,
    bool add_ass,
    char* buf,
    int32_t length
) {
    return llama_chat_apply_template(tmpl, chat, n_msg, add_ass, buf, length);
}
*/
import "C"
import (
	"fmt"
	"runtime"
	"unsafe"
)

// Initialize backend
func BackendInit() {
	C.llama_backend_init()
}

func BackendFree() {
	C.llama_backend_free()
}

// Types
type Token = C.llama_token
type Pos = C.llama_pos
type SeqID = C.llama_seq_id

// Model and Context structures
type Model struct {
	ptr *C.struct_llama_model
}

type Context struct {
	ptr *C.struct_llama_context
}

type Memory struct {
	ptr C.llama_memory_t
}

type Vocab struct {
	ptr *C.struct_llama_vocab
}

type LLama struct {
	Model   *Model
	Context *Context
}

type Batch struct {
	ptr *C.struct_llama_batch
}

type Sampler struct {
	ptr *C.struct_llama_sampler
}

// Model parameters
type ModelParams struct {
	NGPULayers   int32
	SplitMode    SplitMode
	MainGPU      int32
	TensorSplit  []float32
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

// Default parameter functions
func DefaultModelParams() ModelParams {
	cParams := C.llama_model_default_params()
	return ModelParams{
		NGPULayers:   cParams.n_gpu_layers,
		SplitMode:    SplitMode(cParams.split_mode),
		MainGPU:      cParams.main_gpu,
		VocabOnly:    bool(cParams.vocab_only),
		UseMMmap:     bool(cParams.use_mmap),
		UseMLock:     bool(cParams.use_mlock),
		CheckTensors: bool(cParams.check_tensors),
	}
}

func DefaultContextParams() ContextParams {
	cParams := C.llama_context_default_params()
	return ContextParams{
		NCtx:            cParams.n_ctx,
		NBatch:          cParams.n_batch,
		NUBatch:         cParams.n_ubatch,
		NSeqMax:         cParams.n_seq_max,
		NThreads:        cParams.n_threads,
		NThreadsBatch:   cParams.n_threads_batch,
		RopeScalingType: RopeScalingType(cParams.rope_scaling_type),
		PoolingType:     PoolingType(cParams.pooling_type),
		AttentionType:   AttentionType(cParams.attention_type),
		RopeFreqBase:    cParams.rope_freq_base,
		RopeFreqScale:   cParams.rope_freq_scale,
		FlashAttn:       bool(cParams.flash_attn),
		OffloadKQV:      bool(cParams.offload_kqv),
		Embeddings:      bool(cParams.embeddings),
		NoPerf:          bool(cParams.no_perf),
		TypeK:           GGMLType(cParams.type_k),
		TypeV:           GGMLType(cParams.type_v),
	}
}

func DefaultSamplerChainParams() C.struct_llama_sampler_chain_params {
	return C.llama_sampler_chain_default_params()
}

// Model loading functions
func LoadModelFromFile(path string, params ModelParams) (*Model, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	cParams := C.struct_llama_model_params{
		n_gpu_layers:  C.int32_t(params.NGPULayers),
		split_mode:    C.enum_llama_split_mode(params.SplitMode),
		main_gpu:      C.int32_t(params.MainGPU),
		vocab_only:    C.bool(params.VocabOnly),
		use_mmap:      C.bool(params.UseMMmap),
		use_mlock:     C.bool(params.UseMLock),
		check_tensors: C.bool(params.CheckTensors),
	}

	cModel := C.llama_model_load_from_file(cPath, cParams)
	if cModel == nil {
		return nil, fmt.Errorf("failed to load model from file: %s", path)
	}

	model := &Model{ptr: cModel}
	runtime.SetFinalizer(model, (*Model).Free)

	return model, nil
}

func NewContextWithModel(model *Model, params ContextParams) (*Context, error) {
	if model.ptr == nil {
		return nil, fmt.Errorf("model is nil")
	}

	cParams := C.struct_llama_context_params{
		n_ctx:             C.uint32_t(params.NCtx),
		n_batch:           C.uint32_t(params.NBatch),
		n_ubatch:          C.uint32_t(params.NUBatch),
		n_seq_max:         C.uint32_t(params.NSeqMax),
		n_threads:         C.int32_t(params.NThreads),
		n_threads_batch:   C.int32_t(params.NThreadsBatch),
		rope_scaling_type: C.enum_llama_rope_scaling_type(params.RopeScalingType),
		pooling_type:      C.enum_llama_pooling_type(params.PoolingType),
		attention_type:    C.enum_llama_attention_type(params.AttentionType),
		rope_freq_base:    C.float(params.RopeFreqBase),
		rope_freq_scale:   C.float(params.RopeFreqScale),
		flash_attn:        C.bool(params.FlashAttn),
		offload_kqv:       C.bool(params.OffloadKQV),
		embeddings:        C.bool(params.Embeddings),
		no_perf:           C.bool(params.NoPerf),
		type_k:            C.enum_ggml_type(params.TypeK),
		type_v:            C.enum_ggml_type(params.TypeV),
	}

	cCtx := C.llama_init_from_model(model.ptr, cParams)
	if cCtx == nil {
		return nil, fmt.Errorf("failed to create context")
	}

	ctx := &Context{ptr: cCtx}
	runtime.SetFinalizer(ctx, (*Context).Free)

	return ctx, nil
}

// Memory management
func (ctx *Context) Memory() *Memory {
	cMem := C.llama_get_memory(ctx.ptr)
	return &Memory{ptr: cMem}
}

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

func (mem *Memory) CanShift() bool {
	return bool(C.llama_memory_can_shift(mem.ptr))
}

// Vocabulary functions
func (model *Model) Vocab() *Vocab {
	cVocab := C.llama_model_get_vocab(model.ptr)
	return &Vocab{ptr: cVocab}
}

func (vocab *Vocab) GetText(token Token) string {
	cText := C.llama_vocab_get_text(vocab.ptr, C.llama_token(token))
	if cText == nil {
		return ""
	}
	return C.GoString(cText)
}

func (vocab *Vocab) GetScore(token Token) float32 {
	return float32(C.llama_vocab_get_score(vocab.ptr, C.llama_token(token)))
}

func (vocab *Vocab) IsEog(token Token) bool {
	return bool(C.llama_vocab_is_eog(vocab.ptr, C.llama_token(token)))
}

func (vocab *Vocab) IsControl(token Token) bool {
	return bool(C.llama_vocab_is_control(vocab.ptr, C.llama_token(token)))
}

func (vocab *Vocab) BOS() Token {
	return Token(C.llama_vocab_bos(vocab.ptr))
}

func (vocab *Vocab) EOS() Token {
	return Token(C.llama_vocab_eos(vocab.ptr))
}

func (vocab *Vocab) NTokens() int32 {
	return int32(C.llama_vocab_n_tokens(vocab.ptr))
}

// Model information functions
func (model *Model) VocabType() int {
	vocab := C.llama_model_get_vocab(model.ptr)
	return int(C.llama_vocab_type(vocab))
}

func (model *Model) RopeType() int {
	return int(C.llama_model_rope_type(model.ptr))
}

func (model *Model) NCtxTrain() int32 {
	return int32(C.llama_model_n_ctx_train(model.ptr))
}

func (model *Model) NEmbd() int32 {
	return int32(C.llama_model_n_embd(model.ptr))
}

func (model *Model) NLayer() int32 {
	return int32(C.llama_model_n_layer(model.ptr))
}

func (model *Model) NHead() int32 {
	return int32(C.llama_model_n_head(model.ptr))
}

func (model *Model) NHeadKV() int32 {
	return int32(C.llama_model_n_head_kv(model.ptr))
}

func (model *Model) NParams() uint64 {
	return uint64(C.llama_model_n_params(model.ptr))
}

func (model *Model) Size() uint64 {
	return uint64(C.llama_model_size(model.ptr))
}

func (model *Model) HasEncoder() bool {
	return bool(C.llama_model_has_encoder(model.ptr))
}

func (model *Model) HasDecoder() bool {
	return bool(C.llama_model_has_decoder(model.ptr))
}

func (model *Model) IsRecurrent() bool {
	return bool(C.llama_model_is_recurrent(model.ptr))
}

func (model *Model) RopeFreqScaleTrain() float32 {
	return float32(C.llama_model_rope_freq_scale_train(model.ptr))
}

func (model *Model) ChatTemplate(name string) string {
	var cName *C.char
	if name != "" {
		cName = C.CString(name)
		defer C.free(unsafe.Pointer(cName))
	}

	cTemplate := C.llama_model_chat_template(model.ptr, cName)
	if cTemplate == nil {
		return ""
	}
	return C.GoString(cTemplate)
}

// Tokenization functions
func (llm *LLama) Tokenize(text string, addSpecial, parseSpecial bool) []Token {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	vocab := C.llama_model_get_vocab(llm.Model.ptr)

	// First call to get required buffer size
	nTokens := C.llama_tokenize(
		vocab,
		cText,
		C.int32_t(len(text)),
		nil,
		0,
		C.bool(addSpecial),
		C.bool(parseSpecial),
	)

	if nTokens < 0 {
		return nil
	}

	// Allocate buffer and tokenize
	tokens := make([]C.llama_token, nTokens)
	actualTokens := C.llama_tokenize(
		vocab,
		cText,
		C.int32_t(len(text)),
		&tokens[0],
		nTokens,
		C.bool(addSpecial),
		C.bool(parseSpecial),
	)

	if actualTokens < 0 {
		return nil
	}

	// Convert to Go tokens
	result := make([]Token, actualTokens)
	for i := 0; i < int(actualTokens); i++ {
		result[i] = Token(tokens[i])
	}

	return result
}

func (llm *LLama) TokenToPiece(token Token, lstrip int32, special bool) string {
	vocab := C.llama_model_get_vocab(llm.Model.ptr)

	// Get required buffer size
	bufSize := C.llama_token_to_piece(
		vocab,
		C.llama_token(token),
		nil,
		0,
		C.int32_t(lstrip),
		C.bool(special),
	)

	if bufSize < 0 {
		return ""
	}

	// Allocate buffer and get text
	buf := make([]C.char, bufSize+1)
	actualSize := C.llama_token_to_piece(
		vocab,
		C.llama_token(token),
		&buf[0],
		C.int32_t(bufSize),
		C.int32_t(lstrip),
		C.bool(special),
	)

	if actualSize < 0 {
		return ""
	}

	return C.GoStringN(&buf[0], actualSize)
}

func (llm *LLama) Detokenize(tokens []Token, removeSpecial, unparseSpecial bool) string {
	if len(tokens) == 0 {
		return ""
	}

	vocab := C.llama_model_get_vocab(llm.Model.ptr)
	cTokens := make([]C.llama_token, len(tokens))
	for i, token := range tokens {
		cTokens[i] = C.llama_token(token)
	}

	// Get required buffer size
	bufSize := C.llama_detokenize(
		vocab,
		&cTokens[0],
		C.int32_t(len(tokens)),
		nil,
		0,
		C.bool(removeSpecial),
		C.bool(unparseSpecial),
	)

	if bufSize < 0 {
		return ""
	}

	// Allocate buffer and detokenize
	buf := make([]C.char, bufSize+1)
	actualSize := C.llama_detokenize(
		vocab,
		&cTokens[0],
		C.int32_t(len(tokens)),
		&buf[0],
		C.int32_t(bufSize),
		C.bool(removeSpecial),
		C.bool(unparseSpecial),
	)

	if actualSize < 0 {
		return ""
	}

	return C.GoStringN(&buf[0], actualSize)
}

// Chat template application
func ChatApplyTemplate(template string, messages []ChatMessage, addAssistant bool) string {
	if len(messages) == 0 {
		return ""
	}

	// Convert messages to C format
	cMessages := make([]C.struct_llama_chat_message, len(messages))
	var cStrings []*C.char

	for i, msg := range messages {
		cRole := C.CString(msg.Role)
		cContent := C.CString(msg.Content)
		cStrings = append(cStrings, cRole, cContent)

		cMessages[i] = C.struct_llama_chat_message{
			role:    cRole,
			content: cContent,
		}
	}

	// Clean up C strings
	defer func() {
		for _, cStr := range cStrings {
			C.free(unsafe.Pointer(cStr))
		}
	}()

	cTemplate := C.CString(template)
	defer C.free(unsafe.Pointer(cTemplate))

	// Get required buffer size
	bufSize := C.llama_chat_apply_template(
		cTemplate,
		&cMessages[0],
		C.size_t(len(messages)),
		C.bool(addAssistant),
		nil,
		0,
	)

	if bufSize < 0 {
		return ""
	}

	// Apply template
	buf := make([]C.char, bufSize+1)
	actualSize := C.llama_chat_apply_template(
		cTemplate,
		&cMessages[0],
		C.size_t(len(messages)),
		C.bool(addAssistant),
		&buf[0],
		C.int32_t(bufSize),
	)

	if actualSize < 0 {
		return ""
	}

	return C.GoStringN(&buf[0], actualSize)
}

// Batch operations
func NewBatch(nTokens, embd, nSeqMax int32) *Batch {
	cBatch := C.llama_batch_init(
		C.int32_t(nTokens),
		C.int32_t(embd),
		C.int32_t(nSeqMax),
	)

	batch := &Batch{ptr: &cBatch}
	runtime.SetFinalizer(batch, (*Batch).Free)

	return batch
}

func (batch *Batch) Add(token Token, pos Pos, seqIDs []SeqID, logits bool) {
	if batch.ptr.n_tokens >= len(batch.getTokenSlice()) {
		return // Batch is full
	}

	idx := int(batch.ptr.n_tokens)

	// Use helper function to add token
	C.llama_batch_add_wrapper(
		batch.ptr,
		C.int(idx),
		C.llama_token(token),
		C.llama_pos(pos),
		C.llama_seq_id(seqIDs[0]), // Use first sequence ID
		C.bool(logits),
	)

	batch.ptr.n_tokens++
}

func (batch *Batch) getTokenSlice() []Token {
	if batch.ptr.token == nil {
		return nil
	}

	// Create slice from C array
	return (*[1 << 30]Token)(unsafe.Pointer(batch.ptr.token))[:batch.ptr.n_tokens:batch.ptr.n_tokens]
}

func (batch *Batch) Free() {
	if batch.ptr != nil {
		C.llama_batch_free_seq_ids(batch.ptr)
		C.llama_batch_free(*batch.ptr)
		batch.ptr = nil
	}
}

// Inference functions
func (llm *LLama) Encode(batch *Batch) error {
	result := C.llama_encode(llm.Context.ptr, *batch.ptr)
	if result < 0 {
		return fmt.Errorf("encode failed with code: %d", result)
	}
	return nil
}

func (llm *LLama) Decode(batch *Batch) error {
	result := C.llama_decode(llm.Context.ptr, *batch.ptr)
	if result < 0 {
		return fmt.Errorf("decode failed with code: %d", result)
	}
	if result == 1 {
		return fmt.Errorf("could not find KV slot for batch")
	}
	if result == 2 {
		return fmt.Errorf("decode was aborted")
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

// Sampler functions
func NewSamplerChain(params C.struct_llama_sampler_chain_params) *Sampler {
	cSampler := C.llama_sampler_chain_init(params)
	sampler := &Sampler{ptr: cSampler}
	runtime.SetFinalizer(sampler, (*Sampler).Free)
	return sampler
}

func NewSamplerGreedy() *Sampler {
	cSampler := C.llama_sampler_init_greedy()
	sampler := &Sampler{ptr: cSampler}
	runtime.SetFinalizer(sampler, (*Sampler).Free)
	return sampler
}

func NewSamplerDist(seed uint32) *Sampler {
	cSampler := C.llama_sampler_init_dist(C.uint32_t(seed))
	sampler := &Sampler{ptr: cSampler}
	runtime.SetFinalizer(sampler, (*Sampler).Free)
	return sampler
}

func NewSamplerTopK(k int32) *Sampler {
	cSampler := C.llama_sampler_init_top_k(C.int32_t(k))
	sampler := &Sampler{ptr: cSampler}
	runtime.SetFinalizer(sampler, (*Sampler).Free)
	return sampler
}

func NewSamplerTopP(p float32, minKeep int) *Sampler {
	cSampler := C.llama_sampler_init_top_p(C.float(p), C.size_t(minKeep))
	sampler := &Sampler{ptr: cSampler}
	runtime.SetFinalizer(sampler, (*Sampler).Free)
	return sampler
}

func NewSamplerMinP(p float32, minKeep int) *Sampler {
	cSampler := C.llama_sampler_init_min_p(C.float(p), C.size_t(minKeep))
	sampler := &Sampler{ptr: cSampler}
	runtime.SetFinalizer(sampler, (*Sampler).Free)
	return sampler
}

func NewSamplerTemp(temperature float32) *Sampler {
	cSampler := C.llama_sampler_init_temp(C.float(temperature))
	sampler := &Sampler{ptr: cSampler}
	runtime.SetFinalizer(sampler, (*Sampler).Free)
	return sampler
}

func NewSamplerTempExt(temp, delta, exponent float32) *Sampler {
	cSampler := C.llama_sampler_init_temp_ext(C.float(temp), C.float(delta), C.float(exponent))
	sampler := &Sampler{ptr: cSampler}
	runtime.SetFinalizer(sampler, (*Sampler).Free)
	return sampler
}

func NewSamplerMirostat(nVocab int32, seed uint32, tau, eta float32, m int32) *Sampler {
	cSampler := C.llama_sampler_init_mirostat(
		C.int32_t(nVocab),
		C.uint32_t(seed),
		C.float(tau),
		C.float(eta),
		C.int32_t(m),
	)
	sampler := &Sampler{ptr: cSampler}
	runtime.SetFinalizer(sampler, (*Sampler).Free)
	return sampler
}

func NewSamplerMirostatV2(seed uint32, tau, eta float32) *Sampler {
	cSampler := C.llama_sampler_init_mirostat_v2(
		C.uint32_t(seed),
		C.float(tau),
		C.float(eta),
	)
	sampler := &Sampler{ptr: cSampler}
	runtime.SetFinalizer(sampler, (*Sampler).Free)
	return sampler
}

func NewSamplerPenalties(lastN int32, repeat, freq, present float32) *Sampler {
	cSampler := C.llama_sampler_init_penalties(
		C.int32_t(lastN),
		C.float(repeat),
		C.float(freq),
		C.float(present),
	)
	sampler := &Sampler{ptr: cSampler}
	runtime.SetFinalizer(sampler, (*Sampler).Free)
	return sampler
}

func NewSamplerTypical(p float32, minKeep int) *Sampler {
	cSampler := C.llama_sampler_init_typical(C.float(p), C.size_t(minKeep))
	sampler := &Sampler{ptr: cSampler}
	runtime.SetFinalizer(sampler, (*Sampler).Free)
	return sampler
}

func (chain *Sampler) ChainAdd(sampler *Sampler) {
	C.llama_sampler_chain_add(chain.ptr, sampler.ptr)
	// Remove finalizer from added sampler since chain now owns it
	runtime.SetFinalizer(sampler, nil)
	sampler.ptr = nil
}

func (sampler *Sampler) Sample(ctx *Context, idx int32) Token {
	return Token(C.llama_sampler_sample(sampler.ptr, ctx.ptr, C.int32_t(idx)))
}

func (sampler *Sampler) Accept(token Token) {
	C.llama_sampler_accept(sampler.ptr, C.llama_token(token))
}

func (sampler *Sampler) Reset() {
	C.llama_sampler_reset(sampler.ptr)
}

func (sampler *Sampler) GetSeed() uint32 {
	return uint32(C.llama_sampler_get_seed(sampler.ptr))
}

func (sampler *Sampler) Free() {
	if sampler.ptr != nil {
		C.llama_sampler_free(sampler.ptr)
		sampler.ptr = nil
	}
}

// Performance monitoring
func (ctx *Context) PerfContext() PerfContextData {
	cPerf := C.llama_perf_context(ctx.ptr)
	return PerfContextData{
		TStartMs: float64(cPerf.t_start_ms),
		TLoadMs:  float64(cPerf.t_load_ms),
		TPEvalMs: float64(cPerf.t_p_eval_ms),
		TEvalMs:  float64(cPerf.t_eval_ms),
		NPEval:   int32(cPerf.n_p_eval),
		NEval:    int32(cPerf.n_eval),
		NReused:  int32(cPerf.n_reused),
	}
}

func (ctx *Context) PerfReset() {
	C.llama_perf_context_reset(ctx.ptr)
}

// Resource management
func (model *Model) Free() {
	if model.ptr != nil {
		C.llama_model_free(model.ptr)
		model.ptr = nil
	}
}

func (ctx *Context) Free() {
	if ctx.ptr != nil {
		C.llama_free(ctx.ptr)
		ctx.ptr = nil
	}
}

func (llm *LLama) Free() {
	if llm.Context != nil {
		llm.Context.Free()
	}
	if llm.Model != nil {
		llm.Model.Free()
	}
}
