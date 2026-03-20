import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

response = client.chat.completions.create(
    model="gpt-5.4",
    max_completion_tokens=100,
    messages=[{"role": "user", "content": "Twinkle, Twinkle, Little"}]
)

print(response.choices[0].message.content)
