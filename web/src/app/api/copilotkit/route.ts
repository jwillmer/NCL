/**
 * CopilotKit API Route
 *
 * Bridges the Next.js frontend to the Python AG-UI agent.
 * Handles JWT validation and forwards requests to the agent.
 */

import {
  CopilotRuntime,
  ExperimentalEmptyAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import { HttpAgent } from "@ag-ui/client";
import { NextRequest, NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";

const AGENT_URL = process.env.AGENT_URL || "http://localhost:8000/";
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

export const POST = async (req: NextRequest) => {
  // Verify authentication
  if (!(await verifyAuth(req))) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  // Create CopilotKit runtime with HttpAgent pointing to Python agent
  const runtime = new CopilotRuntime({
    agents: {
      default: new HttpAgent({ url: AGENT_URL }),
    },
  });

  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter,
    endpoint: "/api/copilotkit",
  });

  return handleRequest(req);
};
