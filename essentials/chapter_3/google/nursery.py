import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Twinkle, Twinkle, Little",
)

print(response.text)
