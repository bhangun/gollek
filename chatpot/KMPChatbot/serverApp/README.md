# KMP Chatbot Server - Quarkus Backend

A production-ready REST API server built with **Quarkus** for the KMP Chatbot application. Features include REST endpoints, WebSocket support, database persistence, and AI integration.

## 🚀 Features

- **REST API**: Full CRUD operations for conversations and messages
- **WebSocket**: Real-time chat communication
- **Database**: Hibernate ORM with Panache for easy data access
- **AI Integration**: Anthropic Claude API integration
- **Security**: JWT authentication ready (basic auth for demo)
- **Health Checks**: Production-ready health endpoints
- **OpenAPI/Swagger**: Interactive API documentation
- **Fast Startup**: Quarkus dev mode with live reload
- **Native Compilation**: GraalVM native image support

## 📋 Prerequisites

- **JDK 17** or higher
- **Maven 3.8+**
- **Docker** (optional, for PostgreSQL)
- **Anthropic API Key** (for AI features)

## 🔧 Quick Start

### 1. Navigate to Server Directory

```bash
cd serverApp
```

### 2. Configure API Key

Edit `src/main/resources/application.properties`:

```properties
ai.api.key=your-anthropic-api-key-here
```

Or set environment variable:

```bash
export ANTHROPIC_API_KEY=sk-ant-api03-...
```

### 3. Run in Dev Mode

```bash
./mvnw quarkus:dev
```

The server will start at: http://localhost:8080

**Dev Mode Features:**
- Live reload - code changes are immediately applied
- Dev UI at: http://localhost:8080/q/dev/
- Swagger UI at: http://localhost:8080/swagger-ui/
- H2 Console (if enabled)

### 4. Test the API

**REST API:**
```bash
# Send a chat message
curl -X POST http://localhost:8080/api/chat/message \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello, how are you?"}'

# List conversations
curl http://localhost:8080/api/conversations

# Create conversation
curl -X POST http://localhost:8080/api/conversations \
  -H "Content-Type: application/json" \
  -d '{"title": "My First Chat"}'
```

**WebSocket:**
```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8080/ws/chat/1');

ws.onopen = () => {
    console.log('Connected');
    // Send message
    ws.send(JSON.stringify({
        content: 'Hello!',
        conversationId: null
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};
```

## 📁 Project Structure

```
serverApp/
├── pom.xml                          # Maven configuration
├── src/main/
│   ├── java/com/kmpchatbot/server/
│   │   ├── api/                     # REST Controllers
│   │   │   ├── ChatResource.java
│   │   │   └── ConversationResource.java
│   │   ├── api/dto/                 # Data Transfer Objects
│   │   │   ├── MessageRequest.java
│   │   │   ├── MessageResponse.java
│   │   │   └── ChatResponse.java
│   │   ├── domain/                  # JPA Entities
│   │   │   ├── User.java
│   │   │   ├── Conversation.java
│   │   │   └── Message.java
│   │   ├── service/                 # Business Logic
│   │   │   ├── ChatService.java
│   │   │   └── AIService.java
│   │   ├── websocket/               # WebSocket Endpoints
│   │   │   └── ChatWebSocket.java
│   │   └── config/                  # Configuration
│   │       └── DataInitializer.java
│   └── resources/
│       └── application.properties   # Application config
└── README.md
```

## 🗄️ Database

### Development (H2 In-Memory)

By default, the application uses H2 in-memory database:

```properties
quarkus.datasource.db-kind=h2
quarkus.datasource.jdbc.url=jdbc:h2:mem:chatbot
```

Data is reset on each restart - perfect for development!

### Production (PostgreSQL)

For production, uncomment PostgreSQL configuration in `application.properties`:

```properties
%prod.quarkus.datasource.db-kind=postgresql
%prod.quarkus.datasource.jdbc.url=jdbc:postgresql://localhost:5432/chatbot
%prod.quarkus.datasource.username=postgres
%prod.quarkus.datasource.password=postgres
%prod.quarkus.hibernate-orm.database.generation=update
```

**Start PostgreSQL with Docker:**

```bash
docker run --name chatbot-db \
  -e POSTGRES_DB=chatbot \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  -d postgres:15
```

## 🔌 API Endpoints

### Chat Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat/message` | Send a message and get AI response |
| GET | `/api/chat/conversations/{id}/messages` | Get messages in a conversation |
| GET | `/api/chat/health` | Health check |

### Conversation Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/conversations` | List all conversations |
| POST | `/api/conversations` | Create new conversation |
| GET | `/api/conversations/{id}` | Get conversation by ID |
| DELETE | `/api/conversations/{id}` | Delete conversation |
| PUT | `/api/conversations/{id}/archive` | Archive/unarchive conversation |

### WebSocket

| Endpoint | Description |
|----------|-------------|
| `ws://localhost:8080/ws/chat/{userId}` | Real-time chat WebSocket |

## 📊 Database Schema

### Users Table
```sql
CREATE TABLE users (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(255),
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);
```

### Conversations Table
```sql
CREATE TABLE conversations (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    title VARCHAR(255) NOT NULL,
    user_id BIGINT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP,
    is_archived BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

### Messages Table
```sql
CREATE TABLE messages (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    conversation_id BIGINT NOT NULL,
    content TEXT NOT NULL,
    role VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    token_count INTEGER,
    model_used VARCHAR(100),
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);
```

## 🔐 Security

### Default Users

The application creates two default users on startup:

**Demo User:**
- Username: `demo`
- Password: `demo123`
- Email: `demo@kmpchatbot.com`

**Admin User:**
- Username: `admin`
- Password: `admin123`
- Email: `admin@kmpchatbot.com`

⚠️ **WARNING**: Change these in production!

### JWT Authentication (Ready)

The project is configured for JWT but currently uses basic auth for simplicity. To enable JWT:

1. Generate RSA keys:
```bash
openssl genrsa -out privateKey.pem 2048
openssl rsa -in privateKey.pem -pubout -out publicKey.pem
```

2. Place in `src/main/resources/META-INF/resources/`

3. Implement JWT issuance in login endpoint

## 📖 API Documentation

### Swagger UI

Access interactive API docs at: http://localhost:8080/swagger-ui/

### OpenAPI Specification

Get OpenAPI JSON at: http://localhost:8080/q/openapi

### Example Requests

**Send Message:**
```bash
curl -X POST http://localhost:8080/api/chat/message \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Explain quantum computing",
    "conversationId": null
  }'
```

**Create Conversation:**
```bash
curl -X POST http://localhost:8080/api/conversations \
  -H "Content-Type: application/json" \
  -d '{"title": "Quantum Physics Discussion"}'
```

**Get Messages:**
```bash
curl http://localhost:8080/api/chat/conversations/1/messages
```

## 🏗️ Building for Production

### JVM Mode (Recommended)

```bash
./mvnw clean package

# Run the JAR
java -jar target/quarkus-app/quarkus-run.jar
```

### Native Executable (GraalVM)

Requires GraalVM installed:

```bash
./mvnw package -Pnative

# Run native executable
./target/chatbot-server-1.0.0-runner
```

**Native Benefits:**
- Ultra-fast startup (milliseconds)
- Low memory footprint
- Ideal for containers/serverless

### Docker

```bash
# JVM Mode
docker build -f src/main/docker/Dockerfile.jvm -t kmpchatbot-server .
docker run -p 8080:8080 kmpchatbot-server

# Native Mode
docker build -f src/main/docker/Dockerfile.native -t kmpchatbot-server-native .
docker run -p 8080:8080 kmpchatbot-server-native
```

## 🧪 Testing

### Run Tests

```bash
./mvnw test
```

### Test Coverage

```bash
./mvnw verify
```

### Integration Testing

```bash
# Using curl
./test-api.sh

# Using REST Assured (in tests)
./mvnw test -Dtest=ChatResourceTest
```

## ⚙️ Configuration

Key configuration properties:

```properties
# Server
quarkus.http.port=8080
quarkus.http.cors=true

# Database
quarkus.datasource.db-kind=h2
quarkus.hibernate-orm.database.generation=drop-and-create

# AI API
ai.api.base-url=https://api.anthropic.com/v1
ai.api.key=${ANTHROPIC_API_KEY}
ai.api.model=claude-sonnet-4-20250514
ai.api.max-tokens=1024

# Logging
quarkus.log.level=INFO
quarkus.log.category."com.kmpchatbot".level=DEBUG
```

## 🔍 Health Checks

### Liveness

```bash
curl http://localhost:8080/q/health/live
```

### Readiness

```bash
curl http://localhost:8080/q/health/ready
```

### Application Health

```bash
curl http://localhost:8080/api/chat/health
```

## 📈 Monitoring

### Metrics

Access metrics at: http://localhost:8080/q/metrics

### Dev UI

Development dashboard: http://localhost:8080/q/dev/

Includes:
- Configuration editor
- Database browser
- Health checks
- Metrics visualization

## 🚢 Deployment

### Kubernetes/OpenShift

```bash
./mvnw clean package -Dquarkus.kubernetes.deploy=true
```

### Cloud Platforms

- **AWS**: Deploy to ECS, EKS, or Lambda
- **Google Cloud**: Cloud Run or GKE
- **Azure**: Container Instances or AKS
- **Heroku**: Git push deploy

## 🔧 Troubleshooting

**Port Already in Use:**
```bash
# Change port in application.properties
quarkus.http.port=8081
```

**Database Connection Error:**
```bash
# Check PostgreSQL is running
docker ps | grep postgres

# Verify connection string
psql -h localhost -U postgres -d chatbot
```

**AI API Error:**
```bash
# Verify API key is set
echo $ANTHROPIC_API_KEY

# Test API directly
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"claude-sonnet-4-20250514","messages":[{"role":"user","content":"Hi"}],"max_tokens":100}'
```

## 📚 Additional Resources

- [Quarkus Documentation](https://quarkus.io/guides/)
- [Hibernate ORM with Panache](https://quarkus.io/guides/hibernate-orm-panache)
- [RESTEasy Reactive](https://quarkus.io/guides/resteasy-reactive)
- [WebSockets](https://quarkus.io/guides/websockets)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Run tests
5. Submit pull request

## 📄 License

MIT License - see main project LICENSE file

---

**Server is ready!** Start with `./mvnw quarkus:dev` and visit http://localhost:8080/swagger-ui/
