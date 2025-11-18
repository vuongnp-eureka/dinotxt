# Packaging Guide

This document explains how to package and distribute the `dinotxt` library.

## Package Structure

```
dinotxt/
├── setup.py              # Setup configuration
├── pyproject.toml        # Modern Python packaging config
├── MANIFEST.in          # Files to include in distribution
├── requirements.txt     # Runtime dependencies
├── README.md            # Package documentation
├── __init__.py          # Package initialization with version
├── ...                  # Source code
└── weights/             # BPE vocabulary (included in package)
```

## Building the Package

### Build Source Distribution

```bash
cd dinotxt
python setup.py sdist
```

This creates a `.tar.gz` file in `dist/` directory.

### Build Wheel Distribution

```bash
python setup.py bdist_wheel
```

This creates a `.whl` file in `dist/` directory.

### Build Both

```bash
python setup.py sdist bdist_wheel
```

## Installing from Local Build

```bash
pip install dist/dinotxt-0.1.0.tar.gz
# or
pip install dist/dinotxt-0.1.0-py3-none-any.whl
```

## Publishing to PyPI (Optional)

### 1. Create accounts

- PyPI: https://pypi.org/account/register/
- TestPyPI: https://test.pypi.org/account/register/

### 2. Install build tools

```bash
pip install build twine
```

### 3. Build package

```bash
python -m build
```

### 4. Upload to TestPyPI (for testing)

```bash
twine upload --repository testpypi dist/*
```

### 5. Test installation from TestPyPI

```bash
pip install --index-url https://test.pypi.org/simple/ dinotxt
```

### 6. Upload to PyPI (production)

```bash
twine upload dist/*
```

## GitHub Installation

Users can install directly from GitHub:

```bash
pip install git+https://github.com/vuongnp-eureka/dinotxt.git
```

Or from a specific branch/tag:

```bash
pip install git+https://github.com/vuongnp-eureka/dinotxt.git@main
```

## Version Management

Update version in:
1. `dinotxt/__init__.py` - `__version__ = "0.1.0"`
2. `setup.py` - Uses version from `__init__.py`
3. `pyproject.toml` - Update manually if using

## Testing Before Release

1. **Test installation:**
   ```bash
   pip install -e .
   python test_installation.py
   ```

2. **Test imports:**
   ```python
   python -c "import dinotxt; print(dinotxt.__version__)"
   ```

3. **Test functionality:**
   ```bash
   python example_usage.py
   ```

## Checklist Before Publishing

- [ ] Update version number
- [ ] Update README.md with correct GitHub URL
- [ ] Update setup.py with correct author/URL
- [ ] Test installation locally
- [ ] Test imports work correctly
- [ ] Verify all dependencies are listed
- [ ] Check LICENSE file exists
- [ ] Build and test package locally

