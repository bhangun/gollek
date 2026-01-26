package tech.kayys.golek.plugins;

import java.util.ArrayList;
import java.util.List;

import tech.kayys.golek.engine.EnginePlugin;
import tech.kayys.golek.model.ChatMessage;
import tech.kayys.golek.plugin.PluginContext;
import tech.kayys.golek.plugin.PluginException;

public class RAGPlugin implements EnginePlugin {
    private PluginContext context;
    private VectorStore vectorStore;
    private int topK = 3;
    
    @Override
    public String getId() {
        return "rag-plugin";
    }
    
    @Override
    public String getName() {
        return "RAG Plugin";
    }
    
    @Override
    public String getVersion() {
        return "1.0.0";
    }
    
    @Override
    public String getDescription() {
        return "Retrieval Augmented Generation - adds relevant context from knowledge base";
    }
    
    @Override
    public String[] getDependencies() {
        return new String[]{"embeddings-plugin"};
    }
    
    @Override
    public void initialize(PluginContext context) throws PluginException {
        this.context = context;
        
        String vectorStoreType = context.getConfigValue("rag.vector_store", String.class);
        String dataPath = context.getDataDirectory() + "/vectors";
        
        // Initialize vector store (simplified)
        this.vectorStore = new SimpleVectorStore(dataPath);
        
        Integer configTopK = context.getConfigValue("rag.top_k", Integer.class);
        if (configTopK != null) {
            topK = configTopK;
        }
    }
    
    @Override
    public void start() throws PluginException {
        vectorStore.load();
    }
    
    @Override
    public void stop() throws PluginException {
        vectorStore.close();
    }
    
    @Override
    public List<ChatMessage> preprocessChatMessages(List<ChatMessage> messages) {
        if (messages.isEmpty()) {
            return messages;
        }
        
        // Get the last user message
        ChatMessage lastMessage = messages.get(messages.size() - 1);
        if (!"user".equals(lastMessage.role())) {
            return messages;
        }
        
        // Retrieve relevant documents
        List<String> relevantDocs = vectorStore.search(lastMessage.content(), topK);
        
        if (relevantDocs.isEmpty()) {
            return messages;
        }
        
        // Add context to the messages
        String context = "Relevant information:\n" + String.join("\n\n", relevantDocs);
        
        List<ChatMessage> augmented = new ArrayList<>(messages);
        augmented.add(augmented.size() - 1, new ChatMessage("system", context));
        
        return augmented;
    }
    
    // Simplified vector store interface
    private interface VectorStore {
        void load() throws PluginException;
        List<String> search(String query, int topK);
        void close() throws PluginException;
    }
    
    private static class SimpleVectorStore implements VectorStore {
        private final String dataPath;
        private final List<String> documents = new ArrayList<>();
        
        SimpleVectorStore(String dataPath) {
            this.dataPath = dataPath;
        }
        
        @Override
        public void load() {
            // Load documents from disk
        }
        
        @Override
        public List<String> search(String query, int topK) {
            // Simple keyword-based search (in real implementation, use embeddings)
            return documents.stream()
                .filter(doc -> doc.toLowerCase().contains(query.toLowerCase()))
                .limit(topK)
                .toList();
        }
        
        @Override
        public void close() {
            documents.clear();
        }
    }
}
