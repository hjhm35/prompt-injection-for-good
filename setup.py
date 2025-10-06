from setuptools import setup, find_packages

setup(
    name="llm-evaluation",
    version="1.0.0",
    description="Prompt Injection For Good - Prototype Tool for Shadow AI Security Testing",
    author="Tom", 
    packages=find_packages(),
    install_requires=[
        "sqlalchemy>=2.0.0",
        "click>=8.0.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.65.0",
        "aiohttp>=3.8.0",
        "openai>=1.3.0",
        "anthropic>=0.8.0",
        "pandas>=2.0.0",
    ],
    entry_points={
        'console_scripts': [
            'llm-eval=src.main:cli',
        ],
    },
    python_requires=">=3.8",
)