import { GoogleGenAI } from '@google/genai';
import "dotenv/config";

const genAI = new GoogleGenAI(process.env.GOOGLE_API_KEY);

const response = await genAI.models.generateContent({
    model: "gemini-3-flash-preview",
    contents: "Twinkle, Twinkle, Little",
});

console.log(response.text);
