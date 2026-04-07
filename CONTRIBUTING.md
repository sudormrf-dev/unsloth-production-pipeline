# Contributing

Thank you for your interest in contributing to unsloth-production-pipeline!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/unsloth-production-pipeline`
3. Create a virtual environment: `python -m venv .venv && source .venv/bin/activate`
4. Install dev dependencies: `pip install -e ".[dev]"`

## Development Workflow

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make your changes
3. Run linting: `ruff check pipeline/ tests/`
4. Run tests: `pytest tests/ -v`
5. Commit your changes with a clear message
6. Push and open a Pull Request

## Code Style

- Line length: 120 characters (configured in pyproject.toml)
- Use type hints throughout
- Docstrings for all public functions and classes
- Follow the existing module structure

## Pull Request Guidelines

- Include a clear description of what changed and why
- Add tests for new functionality
- Ensure all CI checks pass before requesting review
- Keep PRs focused — one feature or fix per PR

## Reporting Issues

Open a GitHub issue with:
- A clear description of the bug or feature request
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Environment details (Python version, GPU, OS)
