# Contributing to DINOtxt

Thank you for your interest in contributing to DINOtxt!

## Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/vuongnp-eureka/dinotxt.git
   cd dinotxt
   ```

2. **Install in development mode:**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Run tests:**
   ```bash
   python test_installation.py
   pytest tests/  # if tests exist
   ```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints where possible
- Add docstrings to functions and classes
- Keep functions focused and small

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Add tests if applicable
4. Update documentation if needed
5. Ensure all tests pass
6. Submit a pull request with a clear description

## Reporting Issues

When reporting issues, please include:
- Python version
- Operating system
- Error messages and stack traces
- Steps to reproduce
- Expected vs actual behavior

