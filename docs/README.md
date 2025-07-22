# ASR Interface Documentation

This directory contains comprehensive documentation for the ASR Interface project.

## Documentation Structure

### 📖 [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)
**Complete guide for integrating custom ASR backends**

This is the main integration documentation that covers everything you need to know about adding your own ASR models and engines to the system. It includes:

- **Quick Start Guide** - Choose between two integration paths
- **Architecture Overview** - Understanding the system design
- **Path 1: ASRBase + OnlineASRProcessor** - Recommended approach for most use cases
- **Path 2: Complete Custom ASRProcessor** - Advanced approach for full control
- **Testing & Troubleshooting** - Comprehensive testing strategies and common issues
- **Real-world Examples** - Working examples from existing backends

### 🏗️ [ARCHITECTURE.md](ARCHITECTURE.md)
**System architecture and design decisions**

Detailed technical documentation covering:
- Overall system architecture
- Component interactions
- Data flow diagrams
- Design patterns and principles
- Performance considerations

### 🔧 [API_REFERENCE.md](API_REFERENCE.md)
**Complete API documentation**

Comprehensive reference for:
- REST API endpoints
- WebRTC signaling
- Configuration options
- Response formats
- Error codes

### 🚀 [DEPLOYMENT.md](DEPLOYMENT.md)
**Deployment and production setup**

Production deployment guide including:
- Environment setup
- Configuration management
- Performance tuning
- Monitoring and logging
- Security considerations

### 🧪 [TESTING.md](TESTING.md)
**Testing strategies and examples**

Testing documentation covering:
- Unit testing approaches
- Integration testing
- Performance testing
- Test data management
- CI/CD integration

## Supported ASR Backends

- **Whisper Timestamped** (word-level timestamps, non-permissive license)
- **MLX Whisper** (Apple Silicon optimized, word-level timestamps)
- **Whisper (OpenAI official)** (segment-level timestamps, permissive license)

To use the standard OpenAI Whisper backend, set `backend: "whisper"` in your configuration (see below).

### Example: Selecting a Backend

In your configuration (e.g., `ASRConfig`):

```python
backend = "whisper"  # For standard OpenAI Whisper
# or
backend = "mlx_whisper"  # For MLX Whisper
# or
backend = "whisper_timestamped"  # For Whisper Timestamped
```

## Getting Started

1. **New to the project?** Start with the main [README.md](../README.md) for an overview
2. **Want to integrate a custom backend?** Go directly to [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)
3. **Need API details?** Check [API_REFERENCE.md](API_REFERENCE.md)
4. **Deploying to production?** See [DEPLOYMENT.md](DEPLOYMENT.md)

## Contributing to Documentation

When contributing to documentation:

1. **Keep it practical** - Focus on actionable information
2. **Include examples** - Code examples help developers understand quickly
3. **Update consistently** - When code changes, update related docs
4. **Test examples** - Ensure all code examples work as written
5. **Use clear structure** - Consistent formatting and organization

## Documentation Standards

- **Markdown format** - All docs use standard Markdown
- **Code blocks** - Use syntax highlighting for code examples
- **Links** - Use relative links within the docs directory
- **Images** - Store in `docs/images/` if needed
- **Versioning** - Update docs when making breaking changes

---

**Need help?** Check the main [README.md](../README.md) or open an issue on GitHub.
