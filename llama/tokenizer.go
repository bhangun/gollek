package llama

/*
#include "llama.h"
*/
import "C"
import "unsafe"

func (llm *LLama) Tokenize(text string, addSpecial, parseSpecial bool) []Token {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	vocab := C.llama_model_get_vocab(llm.Model.ptr)
	nTokens := C.llama_tokenize(vocab, cText, C.int32_t(len(text)), nil, 0, C.bool(addSpecial), C.bool(parseSpecial))
	if nTokens < 0 {
		return nil
	}

	tokens := make([]C.llama_token, nTokens)
	actual := C.llama_tokenize(vocab, cText, C.int32_t(len(text)), &tokens[0], nTokens, C.bool(addSpecial), C.bool(parseSpecial))
	if actual < 0 {
		return nil
	}

	out := make([]Token, int(actual))
	for i := 0; i < int(actual); i++ {
		out[i] = Token(tokens[i])
	}
	return out
}

func (llm *LLama) TokenToPiece(token Token, lstrip int32, special bool) string {
	vocab := C.llama_model_get_vocab(llm.Model.ptr)
	need := C.llama_token_to_piece(vocab, C.llama_token(token), nil, 0, C.int32_t(lstrip), C.bool(special))
	if need < 0 {
		return ""
	}

	buf := make([]C.char, need+1)
	got := C.llama_token_to_piece(vocab, C.llama_token(token), &buf[0], C.int32_t(need), C.int32_t(lstrip), C.bool(special))
	if got < 0 {
		return ""
	}
	return C.GoStringN(&buf[0], got)
}

func (llm *LLama) Detokenize(tokens []Token, removeSpecial, unparseSpecial bool) string {
	if len(tokens) == 0 {
		return ""
	}
	vocab := C.llama_model_get_vocab(llm.Model.ptr)

	cTokens := make([]C.llama_token, len(tokens))
	for i, t := range tokens {
		cTokens[i] = C.llama_token(t)
	}

	need := C.llama_detokenize(vocab, &cTokens[0], C.int32_t(len(tokens)), nil, 0, C.bool(removeSpecial), C.bool(unparseSpecial))
	if need < 0 {
		return ""
	}

	buf := make([]C.char, need+1)
	got := C.llama_detokenize(vocab, &cTokens[0], C.int32_t(len(tokens)), &buf[0], C.int32_t(need), C.bool(removeSpecial), C.bool(unparseSpecial))
	if got < 0 {
		return ""
	}
	return C.GoStringN(&buf[0], got)
}
