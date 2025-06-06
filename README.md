# LLM CrossCheck AI Service

A robust Python service for cross-checking and validating Large Language Model outputs, designed for enterprise deployment with comprehensive testing, monitoring, and CI/CD pipelines.

## 🏗️ Project Structure

```
llm-crosscheck-ai-svc/
├── 📁 src/                         # Source code
│   └── 📁 llm_crosscheck/
│       ├── 📁 api/                 # FastAPI application
│       ├── 📁 core/                # Core business logic
│       │   ├── logging.py          # Structured logging
│       │   └── prompt_engine.py    # Jinja2 template engine
│       ├── 📁 llms/                # LLM abstraction layer
│       │   ├── base.py             # Abstract LLM interface
│       │   ├── factory.py          # LLM provider factory
│       │   ├── openai_llm.py       # OpenAI implementation
│       │   ├── anthropic_llm.py    # Anthropic implementation
│       │   └── [other providers]   # Additional LLM providers
│       ├── 📁 models/              # Data models and schemas
│       │   └── llm.py              # LLM-specific models
│       ├── 📁 services/            # Service layer
│       │   ├── llm_manager.py      # LLM orchestration
│       │   └── crosscheck_service.py # Cross-checking workflows
│       ├── 📁 utils/               # Utility functions
│       └── 📁 config/              # Configuration management
├── 📁 prompts/                     # Jinja2 prompt templates
│   ├── 📁 system/                  # System prompts
│   ├── 📁 tasks/                   # Task-specific prompts
│   ├── 📁 crosscheck/              # Cross-checking templates
│   └── 📁 common/                  # Reusable components
├── 📁 tests/                       # Test suite
│   ├── 📁 unit/
│   ├── 📁 integration/
│   └── 📁 e2e/
├── 📁 docs/                        # Documentation
├── 📁 scripts/                     # Utility scripts
├── 📁 docker/                      # Docker configurations
├── 📁 .github/                     # GitHub Actions workflows
└── 📁 configs/                     # Environment configurations
```

## 🧠 Architecture Overview

### LLM Abstraction Layer
- **Unified Interface**: Standardised API across multiple LLM providers
- **Provider Support**: OpenAI, Anthropic, Mistral, Llama, Azure OpenAI, and more
- **Error Handling**: Comprehensive error handling with retries and rate limiting
- **Async Operations**: Full async/await support for high performance
- **Factory Pattern**: Easy provider instantiation and configuration

### Prompt Template Engine
- **Jinja2 Powered**: Flexible template system with inheritance and custom filters
- **Template Categories**: Organised by system, tasks, crosscheck, and common templates
- **Variable Validation**: Required and optional variable checking
- **Caching**: Intelligent template caching with hot reload for development
- **Metadata Support**: Template versioning, descriptions, and validation rules

### Service Architecture
- **LLM Manager**: Orchestrates multiple providers and template rendering
- **CrossCheck Service**: High-level workflows for response validation
- **Structured Logging**: JSON-formatted logs with context tracking
- **Health Monitoring**: Comprehensive health checks for all components

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.11+**: Modern Python with type hints and async support
- **FastAPI**: High-performance async web framework
- **Pydantic**: Data validation and serialisation
- **Jinja2**: Template engine for prompt management
- **Structlog**: Structured logging with JSON output

### LLM Providers
- **OpenAI**: GPT-4, GPT-3.5 Turbo models
- **Anthropic**: Claude 3 (Opus, Sonnet, Haiku) models
- **Mistral AI**: Mistral models (when available)
- **Llama**: Local and hosted Llama models
- **Azure OpenAI**: Enterprise OpenAI deployment
- **Extensible**: Easy to add new providers

### Infrastructure
- **Docker**: Containerisation and deployment
- **Redis**: Caching and session management
- **PostgreSQL**: Persistent data storage (when enabled)
- **Prometheus**: Metrics collection and monitoring
- **Grafana**: Metrics visualisation and dashboards

### Development Tools
- **Pytest**: Comprehensive testing framework
- **Black**: Code formatting
- **Ruff**: Fast Python linting
- **MyPy**: Static type checking
- **Pre-commit**: Git hooks for code quality

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Git

### Development Setup

1. **Clone and setup environment:**
   ```bash
   git clone <repository-url>
   cd llm-crosscheck-ai-svc
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   
   pip install -r requirements-dev.txt
   ```

2. **Environment Configuration:**
   ```bash
   cp .env.example .env
   # Edit .env with your configurations
   ```

3. **Run with Docker:**
   ```bash
   docker-compose up --build
   ```

4. **Run locally for development:**
   ```bash
   uvicorn src.llm_crosscheck.api.main:app --reload --host 0.0.0.0 --port 8000
   ```

## 💡 Usage Examples

### Basic LLM Manager Usage

```python
from llm_crosscheck.services.llm_manager import LLMManager
from llm_crosscheck.schemas.llm import LLMProvider, LLMProviderConfig

# Initialise LLM Manager
llm_manager = LLMManager()

# Configure OpenAI provider
openai_config = LLMProviderConfig(
    provider=LLMProvider.OPENAI,
    api_key="your-openai-api-key",
    default_model="gpt-4",
    available_models=["gpt-4", "gpt-3.5-turbo"]
)
llm_manager.register_provider(openai_config)

# Generate response using template
response = await llm_manager.generate_from_template(
    template_name="system/assistant",
    context={
        "assistant_name": "CrossCheck AI",
        "capabilities": ["Response validation", "Code review", "Multi-provider comparison"]
    }
)
print(response.content)
```

### Cross-Checking Responses

```python
from llm_crosscheck.services.crosscheck_service import CrossCheckService

# Initialise CrossCheck service
crosscheck = CrossCheckService()

# Configure multiple providers for comparison
provider_configs = [
    {
        "provider": "openai",
        "api_key": "your-openai-key",
        "default_model": "gpt-4"
    },
    {
        "provider": "anthropic", 
        "api_key": "your-anthropic-key",
        "default_model": "claude-3-sonnet-20240229"
    }
]
crosscheck.configure_providers(provider_configs)

# Validate a response
validation_result = await crosscheck.validate_response(
    query="What are the benefits of renewable energy?",
    response="Renewable energy sources like solar and wind...",
    response_provider="OpenAI GPT-4",
    validation_criteria=["Factual accuracy", "Completeness", "Bias detection"]
)

print(f"Assessment: {validation_result['overall_assessment']}")
print(f"Confidence: {validation_result['confidence_score']}")
```

### Code Review with Templates

```python
# Perform code review using templates
code_review_result = await llm_manager.code_review(
    code="""
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
    """,
    language="python",
    focus_areas=["Performance", "Algorithm efficiency"],
    include_suggestions=True
)

print(code_review_result.content)
```

### Custom Prompt Templates

Create your own templates in the `prompts/` directory:

```jinja2
{# prompts/tasks/my_custom_task.j2 #}
{# description: Custom task template #}
{# required_variables: task_input, context #}

You are an expert in {{ context }}.

Please analyse the following:
{{ task_input }}

Provide your analysis in the following format:
1. **Summary**: Brief overview
2. **Key Points**: Main findings
3. **Recommendations**: Actionable suggestions
```

Then use it:

```python
response = await llm_manager.generate_from_template(
    template_name="tasks/my_custom_task",
    context={
        "task_input": "Your input here",
        "context": "data analysis"
    }
)
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test types
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/
```

## 🐳 Docker

- **Development:** `docker-compose.yml`
- **Production:** `docker-compose.prod.yml`
- **Testing:** `docker-compose.test.yml`

## 📖 API Documentation

Once running, visit:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## 🔧 Development

### Code Quality
- **Linting:** `ruff check`
- **Formatting:** `black . && isort .`
- **Type Checking:** `mypy src/`

### Pre-commit Hooks
```bash
pre-commit install
```

## 🚀 Deployment

Supports deployment to:
- Docker containers
- Kubernetes
- Cloud platforms (AWS, GCP, Azure)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.