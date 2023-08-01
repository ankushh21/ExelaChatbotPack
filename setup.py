
from setuptools import setup, find_packages

setup(
    name = 'ExelaChatbotPack',
    version = '0.1.1',
    author = 'Exela Technologies',
    description = 'This package is used for Leveraging chatbot functinality.',
    license='unlicense',
    packages = find_packages(exclude=['exelachatbotpack_ve']),
    install_requires = [
        'ExelaChatbotPack',
        'langchain==0.0.222',
        'llama_index',
        'openai==0.27.4',
        'tiktoken==0.3.3',
        'python-dotenv',
        'beautifulsoup4',
        'unstructured',
        'pdf2image',
        'docx2txt',
        'chromadb',
        'pypdf',
    ],
    zip_safe=False
)
