# CIAO-Code-In-Architecture-Out

CIAO is a command‑line tool that flattens a code repository with Repomix and then uses the OpenAI API to generate Markdown documentation asynchronously.

## Prerequisites

- Python 3.11 or later installed and available.
- An OpenAI API key set in the environment variable `OPENAI_API_KEY`.
- The `repomix` CLI installed and available on your PATH.

## Installation

1. Clone the repository
2. Create and activate a virtual environment (optional)
3. Install Python dependencies: `pip install argparse asyncio aiofiles tiktoken openai repomix`



## Usage

1. Set your OpenAI API key export `OPENAI_API_KEY="Your API Key"`
2. Set repomix.config.json file to include exclude programming languages
3. Run CIAO on a repository: `python main.py <github repo link or local repo>`


When the run completes successfully, CIAO will:

- Use Repomix (and `repomix.config.json`) to flatten the repository into `full_code.txt`.
- Load documentation templates and guidelines from `prompt.json`.
- Generate documentation sections in parallel and write the final Markdown document to `documentation.txt` in the CIAO project directory.


## Data Folder 
As supplemental material the data folder contains:










