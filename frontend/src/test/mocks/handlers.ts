import { http, HttpResponse } from "msw";

export const handlers = [
  // Mock CopilotKit endpoint
  http.post("*/copilotkit", async () => {
    return HttpResponse.json({
      answer: "Based on the email from John Doe dated January 15, 2024, the project timeline was discussed...",
      sources: [
        {
          file_path: "/data/source/email1.eml",
          document_type: "email",
          email_subject: "Project Update",
          email_initiator: "john.doe@example.com",
          email_participants: ["john.doe@example.com", "jane.smith@example.com"],
          email_date: "2024-01-15T10:30:00Z",
          chunk_content: "The project is on track for completion by Q2...",
          similarity_score: 0.85,
          rerank_score: 0.92,
          heading_path: ["Email Body"],
          root_file_path: "/data/source/email1.eml",
        },
      ],
    });
  }),

  // Mock Supabase Auth
  http.post("*/auth/v1/token", () => {
    return HttpResponse.json({
      access_token: "mock-access-token",
      refresh_token: "mock-refresh-token",
      expires_in: 3600,
      user: {
        id: "mock-user-id",
        email: "test@example.com",
      },
    });
  }),

  // Mock Supabase session
  http.get("*/auth/v1/session", () => {
    return HttpResponse.json({
      access_token: "mock-access-token",
      refresh_token: "mock-refresh-token",
      expires_in: 3600,
      user: {
        id: "mock-user-id",
        email: "test@example.com",
      },
    });
  }),
];
