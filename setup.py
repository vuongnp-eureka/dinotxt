#!/usr/bin/env python3
"""
Setup script for dinotxt package.
"""

from pathlib import Path
import re
from typing import List, Tuple

from setuptools import setup, find_packages


NAME = "dinotxt"
DESCRIPTION = "Minimal DINOtxt library for loading and inferring DINOtxt vision-language models"
URL = "https://github.com/vuongnp-eureka/dinotxt"  # Update with your repo URL
AUTHOR = "vuongnp-eureka"  # Update with your name
REQUIRES_PYTHON = ">=3.8"
HERE = Path(__file__).parent


def get_long_description() -> str:
    """Read long description from README.md"""
    readme_path = HERE / "README.md"
    if readme_path.exists():
        with open(readme_path, encoding="utf-8") as f:
            return f.read()
    return DESCRIPTION


def get_version() -> str:
    """Extract version from __init__.py"""
    init_path = HERE / "dinotxt" / "__init__.py"
    with open(init_path) as f:
        content = f.read()
        match = re.search(r'^__version__ = ["\']([^"\']+)["\']', content, re.MULTILINE)
        if match:
            return match.group(1)
    raise RuntimeError("Unable to find version string")


def get_requirements() -> List[str]:
    """Read requirements from requirements.txt"""
    req_path = HERE / "requirements.txt"
    if not req_path.exists():
        return [
            "torch>=1.13.0",
            "torchvision>=0.14.0",
            "numpy>=1.21.0",
            "Pillow>=8.0.0",
            "ftfy>=6.0.0",
            "regex>=2022.0.0",
        ]
    
    requirements = []
    with open(req_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)
    return requirements


version = get_version()
requirements = get_requirements()

setup(
    name=NAME,
    version=version,
    description=DESCRIPTION,
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author=AUTHOR,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(where=".", exclude=["tests", "*.tests", "*.tests.*", "__pycache__"]),
    package_dir={"": "."},
    package_data={
        "dinotxt": [
            "weights/*.txt.gz",
        ],
    },
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="computer-vision vision-language transformer dinov3 dinotxt",
    project_urls={
        "Bug Reports": f"{URL}/issues",
        "Source": URL,
        "Documentation": f"{URL}#readme",
    },
)

