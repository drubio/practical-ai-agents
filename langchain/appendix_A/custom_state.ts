import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";

// Load environment variables from relative .env file
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
dotenv.config({ path: path.resolve(__dirname, "../../shared/.env") });

// LangChain and LangGraph imports
import { tool } from "@langchain/core/tools";
import { BaseMessage, HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";
import { Annotation, StateGraph, MessagesAnnotation, messagesStateReducer, START, END } from "@langchain/langgraph";
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


// Option 1 - All custom graph state fields
const CustomAgentState = Annotation.Root({
  // Manually define messages channel with its reducer
  messages: Annotation<BaseMessage[]>({
    reducer: messagesStateReducer,
    default: () => [],
  }),
  // Standard channels that overwrite on update
  llmCallCount: Annotation<number>(),
  user: Annotation<string>(),
});


/**
// Option 2 - Equivalent graph state inherting from MessagesAnnotation
const CustomAgentState = Annotation.Root({
    ...MessagesAnnotation.spec,
  llmCallCount: Annotation<number>(),
  user: Annotation<string>(),
});
**/

// 2. Node function implementation
async function callModel(state: typeof CustomAgentState.State) {
  const response = await llm.invoke(state.messages);

  // Safely get llmCallCount in case it was not initialized (default to 0)
  const currentCount = state.llmCallCount ?? 0;
  //Increase llm_call_count after llm.invoke    
  const newCount = currentCount + 1;

  console.log(`Running LLM on behalf of ${state.user}`);
  console.log(`LLM has been run ${newCount} times`);

  // Return updates to state
  return {
    messages: [response],
    llmCallCount: newCount,
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

