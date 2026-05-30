import OpenAI from 'openai';
import "dotenv/config";

const xai = new OpenAI({
  apiKey: process.env.XAI_API_KEY,
  baseURL: "https://api.x.ai/v1",
});

const response = await xai.chat.completions.create({
  model: "grok-4",
  max_completion_tokens: 100,
  messages: [{ role: "user", content: "Twinkle, Twinkle, Little" }],
});

console.log(response.choices[0].message.content);
