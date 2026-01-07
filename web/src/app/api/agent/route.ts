/**
 * AG-UI Agent API Route
 *
 * Proxies requests from the frontend to the Python LangGraph agent.
 * Handles JWT validation before forwarding.
 *
 * This is a simple pass-through proxy - the actual AG-UI protocol
 * handling is done by the HttpAgent in the frontend and the
 * add_langgraph_fastapi_endpoint in the backend.
 */

import { NextRequest, NextResponse } from "next/server";
import { createClient } from "@supabase/supabase-js";

const AGENT_URL = process.env.AGENT_URL || "http://localhost:8000/agent";

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

/**
 * Proxy POST requests to the Python agent.
 * Preserves headers and streams the response back.
 */
export const POST = async (req: NextRequest) => {
  // Verify authentication at gateway level
  if (!(await verifyAuth(req))) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    // Forward the request to the Python agent
    const body = await req.text();

    // Build headers to forward (excluding host-specific headers)
    const headers: HeadersInit = {
      "Content-Type": req.headers.get("content-type") || "application/json",
    };

    // Forward authorization header for defense-in-depth
    const authHeader = req.headers.get("authorization");
    if (authHeader) {
      headers["Authorization"] = authHeader;
    }

    // Forward accept header for SSE streaming
    const acceptHeader = req.headers.get("accept");
    if (acceptHeader) {
      headers["Accept"] = acceptHeader;
    }

    const response = await fetch(AGENT_URL, {
      method: "POST",
      headers,
      body,
    });

    // If response is not OK, return error
    if (!response.ok) {
      const errorText = await response.text();
      console.error("Agent error:", response.status, errorText);
      return new NextResponse(errorText, {
        status: response.status,
        headers: { "Content-Type": "application/json" },
      });
    }

    // Stream the response back to the client
    // This preserves SSE streaming for AG-UI protocol
    const responseHeaders = new Headers();
    response.headers.forEach((value, key) => {
      // Forward relevant headers
      if (
        key.toLowerCase() === "content-type" ||
        key.toLowerCase() === "cache-control" ||
        key.toLowerCase() === "connection"
      ) {
        responseHeaders.set(key, value);
      }
    });

    return new NextResponse(response.body, {
      status: response.status,
      headers: responseHeaders,
    });
  } catch (error) {
    console.error("Proxy error:", error);
    return NextResponse.json(
      { error: "Failed to connect to agent" },
      { status: 502 }
    );
  }
};
