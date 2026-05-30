import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1"
)

response = client.chat.completions.create(
    model="grok-4",
    max_completion_tokens=100,
    messages=[{"role": "user", "content": "Twinkle, Twinkle, Little"}]
)

print(response.choices[0].message.content)
