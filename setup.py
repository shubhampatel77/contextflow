from setuptools import setup, find_packages

setup(
    name="contextflow",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "faiss-cpu>=1.7.4",
        "datasets>=2.12.0"
    ],
    description="Decoder-only RAG implementation with joint optimization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    url="https://github.com/yourusername/contextflow",
)