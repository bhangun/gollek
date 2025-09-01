package llama

import "C"
import (
	"unsafe"
)

func ChatApplyTemplate(template string, messages []ChatMessage, addAssistant bool) string {
	if len(messages) == 0 {
		return ""
	}

	cMessages := make([]C.struct_llama_chat_message, len(messages))
	var cStrings []*C.char

	for i, msg := range messages {
		cRole := C.CString(msg.Role)
		cContent := C.CString(msg.Content)
		cStrings = append(cStrings, cRole, cContent)
		cMessages[i] = C.struct_llama_chat_message{role: cRole, content: cContent}
	}

	defer func() {
		for _, s := range cStrings {
			C.free(unsafe.Pointer(s))
		}
	}()

	cTemplate := C.CString(template)
	defer C.free(unsafe.Pointer(cTemplate))

	need := C.llama_chat_apply_template(cTemplate, &cMessages[0], C.size_t(len(messages)), C.bool(addAssistant), nil, 0)
	if need < 0 {
		return ""
	}

	buf := make([]C.char, need+1)
	got := C.llama_chat_apply_template(cTemplate, &cMessages[0], C.size_t(len(messages)), C.bool(addAssistant), &buf[0], C.int32_t(need))
	if got < 0 {
		return ""
	}
	return C.GoStringN(&buf[0], got)
}
