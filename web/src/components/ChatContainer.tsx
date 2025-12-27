"use client";

/**
 * Chat container component with CopilotKit integration.
 * AI instructions are handled server-side in agent.py - this is a pure UI layer.
 */

import { CopilotChat } from "@copilotkit/react-ui";
import "@copilotkit/react-ui/styles.css";

export function ChatContainer() {
  return (
    <div className="flex-1 flex flex-col h-[calc(100vh-8rem)]">
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
