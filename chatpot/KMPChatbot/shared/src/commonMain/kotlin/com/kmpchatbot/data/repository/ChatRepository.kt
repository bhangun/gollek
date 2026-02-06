package com.kmpchatbot.data.repository

import com.kmpchatbot.data.remote.ChatApi
import com.kmpchatbot.data.remote.ChatApiImpl
import com.kmpchatbot.data.remote.dto.ChatRequest
import com.kmpchatbot.data.remote.dto.MessageDto
import com.kmpchatbot.domain.model.Message
import com.kmpchatbot.domain.model.MessageRole

class ChatRepository(
    private val api: ChatApi = ChatApiImpl()
) {
    
    suspend fun sendMessage(messages: List<Message>): Result<Message> {
        val messageDtos = messages
            .filter { it.role != MessageRole.SYSTEM }
            .map { message ->
                MessageDto(
                    role = when (message.role) {
                        MessageRole.USER -> "user"
                        MessageRole.ASSISTANT -> "assistant"
                        MessageRole.SYSTEM -> "user"
                    },
                    content = message.content
                )
            }
        
        val request = ChatRequest(messages = messageDtos)
        
        return api.sendMessage(request).mapCatching { response ->
            val content = response.content.firstOrNull()?.text ?: "No response"
            Message(
                id = response.id,
                content = content,
                role = MessageRole.ASSISTANT
            )
        }
    }
}
