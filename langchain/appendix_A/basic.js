import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";

// Load environment variables from relative .env file
const __dirname = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.resolve(__dirname, "../../shared/.env") });

// LangChain and LangGraph imports
import { tool } from "@langchain/core/tools";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";
import { StateGraph, MessagesAnnotation, START } from "@langchain/langgraph";
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";
import { randomUUID } from "node:crypto";
import { z } from "zod";

// Define Tool (uses Zod for schema validation)
const generateUUID = tool(
    () => randomUUID(),   
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
async function callAgent(state) {
  const response = await llm.invoke(state.messages);
  return { messages: [response] };
}

// Construct Graph using pre-built MessagesAnnotation state
const builder = new StateGraph(MessagesAnnotation)
  .addNode("agent", callAgent)
  .addNode("tools", new ToolNode(tools))
  .addEdge(START, "agent")
  .addConditionalEdges("agent", toolsCondition)
  .addEdge("tools", "agent");

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
