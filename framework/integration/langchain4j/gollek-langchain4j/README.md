# Gollek SDK :: LangChain4j Integration
 
 `gollek-langchain4j` acts as the bridge connecting the powerful Gollek LLM runtime natively into the Java LangChain4j ecosystem.
 
 ## Capabilities
 - **ChatLanguageModel**: Map `GollekSdk` pipelines to LangChain4j Chat representations (`GollekChatModel`).
 - **StreamingChatLanguageModel**: Asynchronous callback streams bound directly to `gollek-engine` token emit hooks (`GollekStreamingChatModel`).
 - **Embeddings**: Native bridge from the Gollek Vision/Text embeddings pipelines.
 
 ## Architecture Mapping
 This sub-module operates inside the library tier (`gollek-sdk-lib-parent`) and interfaces entirely with `gollek-sdk-ml`, translating generic string buffers to `AiMessage`, `UserMessage`, and `SystemMessage` constructs automatically.
 
 ```java
 import dev.langchain4j.model.chat.ChatLanguageModel;
 import tech.kayys.gollek.ml.Gollek;
 import tech.kayys.gollek.langchain4j.GollekChatModel;
 
 Gollek gnk = Gollek.builder().model("Llama-3").build();
 ChatLanguageModel model = new GollekChatModel(gnk);
 
 String answer = model.generate("Hi, how are you?");
 ```
