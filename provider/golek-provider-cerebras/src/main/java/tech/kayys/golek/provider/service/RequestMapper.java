package tech.kayys.wayang.inference.providers.openai;


public class RequestMapper {


    public OpenAIRequest map(ProviderRequest request) {
        return OpenAIRequest.builder()
            .model(request.getModel())
            .messages(request.getMessages().stream()
                .map(this::mapMessage)
                .collect(Collectors.toList()))
            .temperature(getParameter(request, "temperature", Double.class, 0.7))
            .maxTokens(getParameter(request, "max_tokens", Integer.class, null))
            .topP(getParameter(request, "top_p", Double.class, null))
            .stream(request.isStreaming())
            .build();
    
}