# LLM CrossCheck AI Service - Examples

This directory contains practical examples demonstrating how to use the LLM CrossCheck AI Service.

## 🚀 Getting Started

### Prerequisites

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API keys:**
   ```bash
   # For OpenAI
   export OPENAI_API_KEY="your-openai-api-key"
   
   # For Anthropic
   export ANTHROPIC_API_KEY="your-anthropic-api-key"
   ```

### Running Examples

#### Basic Usage Example
```bash
python examples/basic_usage.py
```

This example demonstrates:
- LLM Manager initialisation
- Provider configuration
- Template-based prompt generation
- Health checks
- Template listing

## 📝 Example Files

### `basic_usage.py`
A comprehensive example showing:
- **Provider Setup**: Configure OpenAI and Anthropic providers
- **Template Usage**: Generate responses using Jinja2 templates
- **Code Review**: Analyse code using the code review template
- **Cross-Checking**: Validate responses between providers
- **Health Monitoring**: Check system and provider health

## 🛠️ Customisation

### Adding Your Own Examples

1. Create a new Python file in this directory
2. Import the necessary modules:
   ```python
   from llm_crosscheck.services.llm_manager import LLMManager
   from llm_crosscheck.schemas.llm import LLMProvider, LLMProviderConfig
   ```
3. Follow the patterns shown in `basic_usage.py`

### Creating Custom Templates

1. Add templates to the `prompts/` directory
2. Use Jinja2 syntax with metadata comments:
   ```jinja2
   {# description: Your template description #}
   {# required_variables: var1, var2 #}
   
   Your template content here with {{ var1 }} and {{ var2 }}
   ```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | Optional* |
| `ANTHROPIC_API_KEY` | Anthropic API key | Optional* |

*At least one provider API key is required

### Provider Models

The examples use cost-effective models by default:
- **OpenAI**: `gpt-3.5-turbo`
- **Anthropic**: `claude-3-haiku-20240307`

You can modify these in the example code to use more powerful models like `gpt-4` or `claude-3-sonnet-20240229`.

## 📚 Next Steps

1. **Explore Templates**: Check the `prompts/` directory for available templates
2. **API Integration**: Use the service in your web applications
3. **Custom Providers**: Extend the system with additional LLM providers
4. **Production Deployment**: Deploy using Docker and the provided configurations

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the project root directory
2. **API Key Errors**: Verify your API keys are set correctly
3. **Template Not Found**: Check that template files exist in the `prompts/` directory
4. **Network Issues**: Ensure you have internet connectivity for API calls

### Getting Help

- Check the main README.md for detailed setup instructions
- Review the API documentation at http://localhost:8000/docs
- Examine the source code in `src/llm_crosscheck/` 