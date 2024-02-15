import os
import sys
import torch
import platform
from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"
QUICK_VERSION = "0.1.0"
PYPI_BUILD = os.getenv("PYPI_BUILD", "0") == "1"
HAS_CUDA = torch.cuda.is_available()

if not PYPI_BUILD and HAS_CUDA:
    try:
        CUDA_VERSION = "".join(os.environ.get("CUDA_VERSION", torch.version.cuda).split("."))[:3]
        QUICK_VERSION += f"+cu{CUDA_VERSION}"
    except Exception as ex:
        raise RuntimeError("Your system must have an Nvidia GPU for installing AutoAWQ")

common_setup_kwargs = {
    "version": QUICK_VERSION,
    "name": "quick",
    "author": "SqueezeBits",
    "license": "MIT",
    "python_requires": ">=3.8.0",
    "description": "QUICK implements a collection of novel CUDA kernels designed for faster inference of weight-only quantized LLMS.",
    "long_description": (Path(__file__).parent / "README.md").read_text(encoding="UTF-8"),
    "long_description_content_type": "text/markdown",
    "url": "https://github.com/SqueezeBits/QUICK",
    "keywords": ["awq", "autoawq", "quick", "quantization", "transformers"],
    "platforms": ["linux", "windows"],
    "classifiers": [
        "Environment :: GPU :: NVIDIA CUDA :: 11.8",
        "Environment :: GPU :: NVIDIA CUDA :: 12",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
    ]
}

requirements = [
    "torch>=2.0.1",
    "transformers>=4.35.0",
    "tokenizers>=0.12.1",
    "accelerate",
    "datasets",
]

# CUDA kernels
if platform.system().lower() != "darwin" and HAS_CUDA:
    requirements.append("autoawq-kernels")

# QUICK kernels
extra_compile_args = {
    "cxx": ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"],
    "nvcc": [
        "-O3",
        "-std=c++17",
        "-DENABLE_BF16",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--threads=8",
    ],
}

extensions = []
if HAS_CUDA:
    extensions.append(
        CUDAExtension(
            "quick_kernels",
            [
                "csrc/pybind.cpp",
                "csrc/gemm_cuda_quick.cu",
            ],
            extra_compile_args=extra_compile_args,
        )
    )

additional_setup_kwargs = {
    "ext_modules": extensions,
    "cmdclass": {"build_ext": BuildExtension},
}

common_setup_kwargs.update(additional_setup_kwargs)

setup(
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "eval": [
            "lm_eval>=0.4.0",
            "tabulate",
            "protobuf",
            "evaluate",
            "scipy"
        ],
    },
    **common_setup_kwargs
)
