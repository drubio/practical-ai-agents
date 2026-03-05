'use client';

import React, { useState } from 'react';
import LangGraphChatPage from './components/chats/LangGraphChatPage';
import LlamaIndexChatPage from './components/chats/LlamaIndexChatPage';
import AssistantUIChatPage from './components/chats/AssistantUIChatPage';
import ReactChatPage from './components/chats/ReactChatPage';

const frameworks = [
  { id: 'langgraph', label: 'LangGraph UI', colorClass: 'bg-blue-600' },
  { id: 'llamaindex', label: 'LlamaIndex UI', colorClass: 'bg-purple-600' },
  { id: 'assistant-ui', label: 'Assistant UI', colorClass: 'bg-green-600' },
  { id: 'react', label: 'React UI', colorClass: 'bg-orange-600' },
] as const;

type FrameworkId = (typeof frameworks)[number]['id'];

export default function Page() {
  // Shared chapter-level boilerplate starts here: tab navigation + framework switching.
  const [activeFramework, setActiveFramework] = useState<FrameworkId>('langgraph');

  return (
    <div className="h-screen">
      <div className="bg-gray-900 p-4 text-white">
        <div className="flex flex-wrap gap-4">
          {frameworks.map((framework) => (
            <button
              key={framework.id}
              onClick={() => setActiveFramework(framework.id)}
              className={`rounded px-4 py-2 transition-colors ${
                activeFramework === framework.id ? framework.colorClass : 'bg-gray-700 hover:bg-gray-600'
              }`}
            >
              {framework.label}
            </button>
          ))}
        </div>
      </div>

      {/* Framework-specific logic starts in dedicated component files below. */}
      {activeFramework === 'langgraph' && <LangGraphChatPage />}
      {activeFramework === 'llamaindex' && <LlamaIndexChatPage />}
      {activeFramework === 'assistant-ui' && <AssistantUIChatPage />}
      {activeFramework === 'react' && <ReactChatPage />}
    </div>
  );
}
