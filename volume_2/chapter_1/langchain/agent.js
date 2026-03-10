#!/usr/bin/env node

import readline from "node:readline";
import * as z from "../node_modules/zod";
import { createAgent, tool } from "../node_modules/langchain";

import {
  summarizeText,
  extractKeywords,
  extractTasks,
  scorePriority,
  routeWorkflow,
  parseContent,
  resolveDatetime,
  formatJson,
  calculator,
  analyzeText
} from "../tools.js";


function logTool(name, fn) {
  return (input) => {

    console.log("\n===== LOCAL TOOL CALL =====");
    console.log("Tool:", name);
    console.log("Input:", input);

    const result = fn(input);

    console.log("\n===== TOOL RESULT =====");
    console.log(result);

    return result;
  };
}


const tools = [

tool(
  logTool("summarize_text", ({ text }) => summarizeText(text)),
  { name:"summarize_text", schema:z.object({text:z.string()}) }
),

tool(
  logTool("extract_keywords", ({ text }) => extractKeywords(text)),
  { name:"extract_keywords", schema:z.object({text:z.string()}) }
),

tool(
  logTool("extract_tasks", ({ text }) => extractTasks(text)),
  { name:"extract_tasks", schema:z.object({text:z.string()}) }
),

tool(
  logTool("score_priority", ({ text }) => scorePriority(text)),
  { name:"score_priority", schema:z.object({text:z.string()}) }
),

tool(
  logTool("route_workflow", ({ text }) => routeWorkflow(text)),
  { name:"route_workflow", schema:z.object({text:z.string()}) }
),

tool(
  logTool("parse_content", ({ content }) => parseContent(content)),
  { name:"parse_content", schema:z.object({content:z.string()}) }
),

tool(
  logTool("resolve_datetime", ({ text }) => resolveDatetime(text)),
  { name:"resolve_datetime", schema:z.object({text:z.string()}) }
),

tool(
  logTool("format_json", ({ input }) => formatJson(input)),
  { name:"format_json", schema:z.object({input:z.any()}) }
),

tool(
  logTool("calculator", ({ expression }) => calculator(expression)),
  { name:"calculator", schema:z.object({expression:z.string()}) }
),

tool(
  logTool("analyze_text", ({ text }) => analyzeText(text)),
  { name:"analyze_text", schema:z.object({text:z.string()}) }
)

];


const agent = createAgent({
  model:"gpt-4.1",
  tools,
  systemPrompt:`

You are an AI assistant that can use tools.

Reason step-by-step using this format:

Thought: reasoning about the problem
Action: tool name
Action Input: JSON input for the tool
Observation: tool result

Repeat until the answer is ready.

Finish with:

Final Answer: the response for the user.
`
});


console.log("\n===== AGENT CLI =====\n");

const rl = readline.createInterface({
  input:process.stdin,
  output:process.stdout
});


function ask(){

  rl.question("> ", async input => {

    console.log("\n===== USER INPUT =====");
    console.log(input);

    const result = await agent.invoke({
      messages:[{role:"user",content:input}]
    });

    console.log("\n===== FINAL ANSWER =====");
    console.log(result.output);
    console.log("");

    ask();

  });

}

ask();
