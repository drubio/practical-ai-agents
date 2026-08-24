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
import { StateGraph, MessagesAnnotation, START, END } from "@langchain/langgraph";
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


import { Annotation, messagesStateReducer } from "@langchain/langgraph";
import { BaseMessage } from "@langchain/core/messages";


// Long explicit state definition with NO MessagesAnnotation inheritance
const CustomAgentState = Annotation.Root({
  // Manually define messages channel with its reducer
  messages: Annotation({
    reducer: messagesStateReducer,
    default: () => [],
  }),
  // Reducer for llmCallCount
  llmCallCount: Annotation({
    reducer: (current, update) => current + update,
    default: () => 0,
  }),
  user: Annotation(),
});

/**
// Extending MessagesAnnotation automatically includes the merged "messages" key
const CustomAgentState = Annotation.Root({
    ...MessagesAnnotation.spec,
   llmCallCount: Annotation({
   reducer: (current, update) => current + update,
    default: () => 0,
  }),
  user: Annotation(),
});
**/

// 2. Node function implementation
async function callModel(state) {
  const response = await llm.invoke(state.messages);

  console.log(`Running LLM on behalf of ${state.user}`);
  console.log(`LLM has been run ${state.llmCallCount + 1} times`);

  // Return updates to state
  return {
    messages: [response],
    llmCallCount: 1,
  };
}


// Construct Graph 
const builder = new StateGraph(CustomAgentState)
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
  llmCallCount : 0,
  user: "drubio"
});

// Output final response
const lastMessage = result.messages[result.messages.length - 1];
console.log(lastMessage.content);

