package com.kmpchatbot.presentation

import com.kmpchatbot.data.repository.ChatRepository
import com.kmpchatbot.domain.model.ChatState
import com.kmpchatbot.domain.model.Message
import com.kmpchatbot.domain.model.MessageRole
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlin.random.Random

class ChatViewModel(
    private val repository: ChatRepository = ChatRepository(),
    private val scope: CoroutineScope
) {
    
    private val _state = MutableStateFlow(ChatState())
    val state: StateFlow<ChatState> = _state.asStateFlow()
    
    fun onInputChanged(input: String) {
        _state.update { it.copy(currentInput = input) }
    }
    
    fun sendMessage() {
        val input = _state.value.currentInput.trim()
        if (input.isEmpty() || _state.value.isLoading) return
        
        val userMessage = Message(
            id = generateId(),
            content = input,
            role = MessageRole.USER
        )
        
        _state.update { 
            it.copy(
                messages = it.messages + userMessage,
                currentInput = "",
                isLoading = true,
                error = null
            )
        }
        
        scope.launch {
            repository.sendMessage(_state.value.messages).fold(
                onSuccess = { assistantMessage ->
                    _state.update { 
                        it.copy(
                            messages = it.messages + assistantMessage,
                            isLoading = false
                        )
                    }
                },
                onFailure = { error ->
                    _state.update { 
                        it.copy(
                            isLoading = false,
                            error = error.message ?: "Unknown error occurred"
                        )
                    }
                }
            )
        }
    }
    
    fun clearError() {
        _state.update { it.copy(error = null) }
    }
    
    fun clearChat() {
        _state.update { ChatState() }
    }
    
    private fun generateId(): String {
        return "msg_${Random.nextLong()}"
    }
}
