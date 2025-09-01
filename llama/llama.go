package llama

import "C"

// Backend init/free
func BackendInit() { C.llama_backend_init() }
func BackendFree() { C.llama_backend_free() }
