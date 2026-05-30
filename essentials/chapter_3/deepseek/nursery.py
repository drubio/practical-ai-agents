import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    max_completion_tokens=100,
    messages=[{"role": "user", "content": "Twinkle, Twinkle, Little"}]
)

print(response.choices[0].message.content)
