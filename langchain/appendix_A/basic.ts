import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";

// Load environment variables from relative .env file
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
dotenv.config({ path: path.resolve(__dirname, "../../shared/.env") });

// LangChain and LangGraph imports
import { tool } from "@langchain/core/tools";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";
import { StateGraph, MessagesAnnotation, START, END } from "@langchain/langgraph";
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";
import { randomUUID } from "node:crypto";
import { z } from "zod";

// Define Tool (uses Zod for schema validation)
const generateUUID = tool(
    async () => randomUUID(),
    {
      name: "generate_uuid",
      description: "Generate a unique UUID",
      schema: z.object({}), // no arguments  
    }
);

const tools = [generateUUID];
const llm = new ChatOpenAI({
    model: "gpt-5.6-luna",
    modelKwargs: { reasoning_effort: "none"}
}).bindTools(tools);

// Node function to unpack state.messages for the LLM
async function callModel(state: typeof MessagesAnnotation.State) {
  const response = await llm.invoke(state.messages);
  return { messages: [response] };
}

// Construct Graph 
const builder = new StateGraph(MessagesAnnotation)
  .addNode("model", callModel)
  .addNode("tools", new ToolNode(tools))
  .addEdge(START, "model")
  .addConditionalEdges("model", toolsCondition)
  .addEdge("tools", "model")
  .addEdge("model", END);

const graph = builder.compile();

// Invoke Graph
const result = await graph.invoke({
  messages: [
    new SystemMessage(
      "Use generate_uuid when user asks for UUID. Keep responses short."
    ),
    new HumanMessage("Generate a ticket ID for last night's network outage"),
  ],
});

// Output final response
const lastMessage = result.messages[result.messages.length - 1];
console.log(lastMessage.content);
