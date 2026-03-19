import OpenAI from 'openai';
import "dotenv/config";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const response = await client.chat.completions.create({
  model: "gpt-5.2",
  max_completion_tokens: 100,
  messages: [{ role: "user", content: "Twinkle, Twinkle, Little" }],
});

console.log(response.choices[0].message.content);
