# Document GPT

LLM chatbot based on your PDF document built with Langchain and Gradio

## Requirements

- [Git](https://git-scm.com/)
- Python 3.8+ and pip. Use either [pyenv](https://github.com/pyenv/pyenv) or [conda](https://docs.conda.io/en/latest/)

## Instructions

1. Clone this repo by running `git clone https://github.com/veronicaeyo/document-gpt.git`
2. Change your directory to document-gpt by running: `cd document-gpt`
3. Copy the `.env.sample` to `.env` and replace the `OPENAI_API_KEY` in .env with your own API Key
4. Create and activate a virtual environment using [pyenv](https://akrabat.com/creating-virtual-environments-with-pyenv/) or [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
5. Install dependencies using `pip install -r requiremnts.txt`
6. Run `gradio app.py` to start the application, app should be running on `http://127.0.0.1:7860`

## Demo

![demo](images/demo.gif)
