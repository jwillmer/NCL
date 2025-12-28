"use client";

/**
 * Chat container component with CopilotKit integration.
 * AI instructions are handled server-side in agent.py - this is a pure UI layer.
 *
 * State sharing is set up between Python backend and frontend via LangGraph protocol.
 * Progress indicators are rendered via useCoAgentStateRender hook, which receives
 * real-time updates from copilotkit_emit_state() calls in the Python agent.
 */

import { CopilotChat } from "@copilotkit/react-ui";
import { useCoAgent, useCoAgentStateRender } from "@copilotkit/react-core";
import "@copilotkit/react-ui/styles.css";
import { RAGState, initialRAGState } from "@/types/rag";

function SearchProgress({ message }: { message: string }) {
  return (
    <div className="flex items-center gap-2 px-4 py-3 bg-blue-50 border border-blue-200 rounded-lg mx-4 my-2">
      <div className="animate-spin h-4 w-4 border-2 border-blue-500 border-t-transparent rounded-full" />
      <span className="text-sm text-blue-700">{message}</span>
    </div>
  );
}

export function ChatContainer() {
  const { state } = useCoAgent<RAGState>({
    name: "default",
    initialState: initialRAGState,
  });

  // Render search progress during agent execution
  useCoAgentStateRender<RAGState>({
    name: "default",
    render: ({ state }) => {
      if (state.search_progress) {
        return <SearchProgress message={state.search_progress} />;
      }
      return null;
    },
  });

  return (
    <div className="flex-1 flex flex-col h-[calc(100vh-8rem)]">
      {state.error_message && (
        <div className="px-4 py-2 bg-red-50 border-b border-red-200">
          <div className="text-sm text-red-700">
            Error: {state.error_message}
          </div>
        </div>
      )}
      <CopilotChat
        labels={{
          title: "MTSS Assistant",
          initial: "Hello! I can help you find solutions to technical issues on your vessel. Ask me about past maintenance problems, equipment failures, or search our knowledge base for technical documentation and procedures.",
          placeholder: "Describe an issue or search for technical information...",
        }}
        className="flex-1 [&_.copilotKitChat]:h-full [&_.copilotKitMessages]:max-h-[calc(100vh-16rem)]"
      />
    </div>
  );
}
