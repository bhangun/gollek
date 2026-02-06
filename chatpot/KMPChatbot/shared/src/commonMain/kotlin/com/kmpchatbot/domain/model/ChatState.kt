package com.kmpchatbot.domain.model

data class ChatState(
    val messages: List<Message> = emptyList(),
    val isLoading: Boolean = false,
    val error: String? = null,
    val currentInput: String = ""
)
