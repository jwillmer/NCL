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
          title: "NCL Email Assistant",
          initial: "Hello! I can help you search through your email archive. Ask me anything about your emails and attachments.",
          placeholder: "Ask about your emails...",
        }}
        className="flex-1 [&_.copilotKitChat]:h-full [&_.copilotKitMessages]:max-h-[calc(100vh-16rem)]"
      />
    </div>
  );
}
