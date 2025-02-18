from setuptools import setup, find_packages

setup(
    name="sharknet-pipeline",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.48.2",
        "unsloth>=2025.2.4",
        "vllm>=0.7.1",
        "tqdm>=4.65.0",
        "pillow>=10.0.0"
    ],
) 