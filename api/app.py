from fastapi import FastAPI
from langserve import add_routes
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from dotenv import load_dotenv
import uvicorn
import os

load_dotenv()
# Ensure the key is set before constructing the model
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple API Server",
)

# OpenAI chat model (new import path)
model = ChatOpenAI(model="gpt-3.5-turbo")  # pick your model

# Optional: Ollama local model
# llm = Ollama(model="llama2")

prompt1 = ChatPromptTemplate.from_template(
    "Write me an essay about {topic} with 100 words"
)
prompt2 = ChatPromptTemplate.from_template(
    "Write me a poem about {topic} for a 5-year-old child with 100 words"
)

# Expose raw model (OpenAI)
add_routes(app, model, path="/openai")

# Expose chains
add_routes(app, prompt1 | model, path="/essay")
add_routes(app, prompt2 | model, path="/poem")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
