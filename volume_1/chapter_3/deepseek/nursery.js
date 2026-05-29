import OpenAI from 'openai';
import "dotenv/config";

const deepseek = new OpenAI({
  apiKey: process.env.DEEPSEEK_API_KEY,
  baseURL: "https://api.deepseek.com",
});

const response = await deepseek.chat.completions.create({
  model: "deepseek-v4-flash",
  max_completion_tokens: 100,
  messages: [{ role: "user", content: "Twinkle, Twinkle, Little" }],
});

console.log(response.choices[0].message.content);
