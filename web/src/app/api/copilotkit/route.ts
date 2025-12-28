/**
 * CopilotKit API Route
 *
 * Bridges the Next.js frontend to the Python LangGraph agent.
 * Handles JWT validation and forwards requests to the agent.
 *
 * Note: The CopilotKit runtime automatically forwards Authorization headers
 * to HttpAgent instances. We only need to validate the token here.
 */

import {
  CopilotRuntime,
  ExperimentalEmptyAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import { HttpAgent } from "@ag-ui/client";
import { NextRequest, NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";

const AGENT_URL = process.env.AGENT_URL || "http://localhost:8000/copilotkit";
const serviceAdapter = new ExperimentalEmptyAdapter();

/**
 * Verify Supabase JWT token from request.
 */
async function verifyAuth(req: NextRequest): Promise<boolean> {
  const authHeader = req.headers.get("authorization");
  if (!authHeader?.startsWith("Bearer ")) {
    return false;
  }

  const token = authHeader.slice(7);

  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

  if (!supabaseUrl || !supabaseKey) {
    console.error("Missing Supabase configuration");
    return false;
  }

  const supabase = createClient(supabaseUrl, supabaseKey);
  const { data: { user }, error } = await supabase.auth.getUser(token);

  if (error) {
    console.error("Auth error:", error.message);
    return false;
  }

  return !!user;
}

// Create runtime with HttpAgent - the LangGraph agent exposes AG-UI protocol
// via add_langgraph_fastapi_endpoint, so we use HttpAgent to communicate with it
const runtime = new CopilotRuntime({
  agents: {
    default: new HttpAgent({
      url: AGENT_URL,
    }),
  },
});

export const POST = async (req: NextRequest) => {
  // Verify authentication at gateway level
  if (!(await verifyAuth(req))) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  // CopilotKit runtime automatically forwards Authorization header to the agent
  // (defense-in-depth: Python backend also validates the token)
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter,
    endpoint: "/api/copilotkit",
  });

  return handleRequest(req);
};
