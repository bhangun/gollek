import SwiftUI
import shared

@main
struct iOSApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

struct ContentView: View {
    @StateObject private var viewModel = ChatViewModelWrapper()
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Messages list
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 12) {
                            ForEach(viewModel.messages, id: \.id) { message in
                                MessageBubble(
                                    message: message,
                                    isUser: message.role == .user
                                )
                                .id(message.id)
                            }
                            
                            if viewModel.isLoading {
                                HStack {
                                    ProgressView()
                                        .padding()
                                    Spacer()
                                }
                            }
                        }
                        .padding()
                    }
                    .onChange(of: viewModel.messages.count) { _ in
                        if let lastMessage = viewModel.messages.last {
                            withAnimation {
                                proxy.scrollTo(lastMessage.id, anchor: .bottom)
                            }
                        }
                    }
                }
                
                // Error message
                if let error = viewModel.error {
                    HStack {
                        Text(error)
                            .font(.caption)
                            .foregroundColor(.white)
                            .padding()
                        
                        Spacer()
                        
                        Button("Dismiss") {
                            viewModel.clearError()
                        }
                        .foregroundColor(.white)
                        .padding(.trailing)
                    }
                    .background(Color.red)
                }
                
                // Input field
                HStack(spacing: 12) {
                    TextField("Type a message...", text: $viewModel.currentInput, axis: .vertical)
                        .textFieldStyle(.roundedBorder)
                        .lineLimit(1...4)
                        .disabled(viewModel.isLoading)
                    
                    Button(action: {
                        viewModel.sendMessage()
                    }) {
                        Image(systemName: "paperplane.fill")
                            .foregroundColor(.white)
                            .padding(12)
                            .background(
                                viewModel.currentInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || viewModel.isLoading
                                ? Color.gray
                                : Color.blue
                            )
                            .clipShape(Circle())
                    }
                    .disabled(viewModel.currentInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || viewModel.isLoading)
                }
                .padding()
                .background(Color(uiColor: .systemBackground))
                .shadow(radius: 2)
            }
            .navigationTitle("KMP Chatbot")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: {
                        viewModel.clearChat()
                    }) {
                        Image(systemName: "trash")
                    }
                }
            }
        }
    }
}

struct MessageBubble: View {
    let message: Message
    let isUser: Bool
    
    var body: some View {
        HStack {
            if isUser { Spacer() }
            
            Text(message.content)
                .padding(12)
                .background(isUser ? Color.blue : Color(uiColor: .systemGray5))
                .foregroundColor(isUser ? .white : .primary)
                .cornerRadius(16)
                .frame(maxWidth: 320, alignment: isUser ? .trailing : .leading)
            
            if !isUser { Spacer() }
        }
    }
}

// ViewModel wrapper for SwiftUI
class ChatViewModelWrapper: ObservableObject {
    private let viewModel: ChatViewModel
    
    @Published var messages: [Message] = []
    @Published var currentInput: String = ""
    @Published var isLoading: Bool = false
    @Published var error: String? = nil
    
    init() {
        viewModel = ChatViewModel(
            repository: ChatRepository(),
            scope: CoroutineScopeKt.MainScope()
        )
        
        // Observe state changes
        viewModel.state.subscribe { state in
            if let state = state {
                DispatchQueue.main.async {
                    self.messages = state.messages
                    self.currentInput = state.currentInput
                    self.isLoading = state.isLoading
                    self.error = state.error
                }
            }
        }
    }
    
    func sendMessage() {
        viewModel.sendMessage()
    }
    
    func clearError() {
        viewModel.clearError()
    }
    
    func clearChat() {
        viewModel.clearChat()
    }
}

// Extension to observe Kotlin Flow
extension Kotlinx_coroutines_coreStateFlow {
    func subscribe(onValue: @escaping (T?) -> Void) {
        let scope = CoroutineScopeKt.MainScope()
        
        self.collect(collector: FlowCollector { value in
            onValue(value)
        }, completionHandler: { error in
            if let error = error {
                print("Flow error: \(error)")
            }
        })
    }
}

class FlowCollector<T>: Kotlinx_coroutines_coreFlowCollector {
    let callback: (T?) -> Void
    
    init(callback: @escaping (T?) -> Void) {
        self.callback = callback
    }
    
    func emit(value: Any?, completionHandler: @escaping (Error?) -> Void) {
        callback(value as? T)
        completionHandler(nil)
    }
}

// Helper to create MainScope
class CoroutineScopeKt {
    static func MainScope() -> Kotlinx_coroutines_coreCoroutineScope {
        return Kotlinx_coroutines_coreCoroutineScopeKt.MainScope()
    }
}
