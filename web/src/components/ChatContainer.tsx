"use client";

/**
 * Chat container component with CopilotKit integration.
 * AI instructions are handled server-side in agent.py - this is a pure UI layer.
 *
 * State sharing is set up between Python backend and frontend via AG-UI protocol.
 * Currently shows error states; progress indicators require intermediate state
 * emission which pydantic-ai doesn't support yet (ToolReturn only emits on completion).
 */

import { CopilotChat } from "@copilotkit/react-ui";
import { useAgent } from "@copilotkit/react-core/v2";
import "@copilotkit/react-ui/styles.css";
import { RAGState, initialRAGState } from "@/types/rag";

export function ChatContainer() {
  const { agent } = useAgent({ agentId: "default" });
  const state: RAGState = { ...initialRAGState, ...agent.state };

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
