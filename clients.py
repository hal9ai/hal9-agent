from groq import Groq
import os

hal9_url = os.environ["HAL9_URL"]  # e.g. "https://api.hal9.com"

DEFAULT_MODEL = "qwen/qwen3.6-27b"

groq_client = Groq(
    base_url=f"{hal9_url}/proxy/server=https://api.groq.com/",
    api_key=os.environ["HAL9_TOKEN"],
)
