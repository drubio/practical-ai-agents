import Anthropic from '@anthropic-ai/sdk';
import "dotenv/config";

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

const response = await anthropic.messages.create({
  model: "claude-sonnet-4-5",
  max_tokens: 100,
  messages: [{ role: "user", content: "Twinkle, Twinkle, Little" }],
});

console.log(response.content[0].text);
