package llama

/*
#include "llama.h"
*/
import "C"

func (vocab *Vocab) GetText(token Token) string {
	c := C.llama_vocab_get_text(vocab.ptr, C.llama_token(token))
	if c == nil {
		return ""
	}
	return C.GoString(c)
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
func (vocab *Vocab) BOS() Token     { return Token(C.llama_vocab_bos(vocab.ptr)) }
func (vocab *Vocab) EOS() Token     { return Token(C.llama_vocab_eos(vocab.ptr)) }
func (vocab *Vocab) NTokens() int32 { return int32(C.llama_vocab_n_tokens(vocab.ptr)) }
