from openai import OpenAI
import os

hal9_url = os.environ["HAL9_URL"]  # e.g. "https://api.hal9.com"

# OpenAI Client
openai_client = OpenAI(
    base_url=f"{hal9_url}/proxy/server=https://api.openai.com/v1/",
    api_key=os.environ["HAL9_TOKEN"]
)

# Groq Client
groq_client = OpenAI(
    base_url=f"{hal9_url}/proxy/server=https://api.groq.com/openai/v1",
    api_key=os.environ["HAL9_TOKEN"]
)