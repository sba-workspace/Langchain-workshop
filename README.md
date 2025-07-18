# LangChain Crash Course

Welcome to the LangChain Crash Course! This repository contains hands-on examples and tutorials to help you master LangChain, a powerful framework for building applications with language models.

## 📋 Prerequisites

- Python 3.10 - 3.12.2
- [Poetry](https://python-poetry.org/) for dependency management
- API keys for the LLM providers you plan to use (OpenAI, Google, Groq, etc.)

## 🚀 Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd langchain-crash-course
   ```

2. Copy the example environment file and update with your API keys:
   ```bash
   cp example.env .env
   # Edit .env with your API keys
   ```

3. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

4. Activate the virtual environment:
   ```bash
   poetry shell
   ```

## 📚 Course Structure

The course is organized into the following sections:

### 1. Chat Models
Introduction to different LLM providers and how to use them with LangChain.

### 2. Prompt Templates
Learn how to create reusable prompt templates and manage prompts effectively.

### 3. Chains
Explore different types of chains for building complex LLM applications.

### 4. RAG (Retrieval-Augmented Generation)
Implement advanced RAG applications with various retrieval and generation techniques.

## 🛠️ Features

- Support for multiple LLM providers (OpenAI, Google, Ollama, Groq)
- Practical examples of LangChain components
- RAG implementation with Chroma and other vector stores
- Integration with various tools and utilities

## 🔑 Environment Variables

Create a `.env` file in the root directory with the following variables:

```
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Google
GOOGLE_API_KEY=your_google_api_key

# Groq
GROQ_API_KEY=your_groq_api_key

# Other providers...
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain Documentation](https://python.langchain.com/)
- [LangChain Community](https://github.com/langchain-ai/langchain)

---

Happy coding! 🚀