/**
 * Citations API Proxy Route
 *
 * Proxies citation requests to the Python backend.
 * Supports two auth methods:
 * 1. Authorization header (for programmatic API calls)
 * 2. Supabase session cookie (for browser navigation)
 */

import { NextRequest, NextResponse } from "next/server";
import { createServerClient } from "@supabase/ssr";
import { cookies } from "next/headers";

const BACKEND_URL = (process.env.AGENT_URL || "http://localhost:8000/copilotkit").replace(
  "/copilotkit",
  ""
);

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ chunkId: string }> }
) {
  const { chunkId } = await params;

  // Try Authorization header first
  let authHeader = req.headers.get("authorization");

  // Fall back to Supabase session cookie
  if (!authHeader?.startsWith("Bearer ")) {
    try {
      const cookieStore = await cookies();
      const supabase = createServerClient(
        process.env.NEXT_PUBLIC_SUPABASE_URL!,
        process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
        {
          cookies: {
            getAll: () => cookieStore.getAll(),
          },
        }
      );
      const { data: { user } } = await supabase.auth.getUser();
      if (user) {
        const { data: { session } } = await supabase.auth.getSession();
        if (session?.access_token) {
          authHeader = `Bearer ${session.access_token}`;
        }
      }
    } catch (error) {
      console.error("Failed to get session from cookie:", error);
    }
  }

  if (!authHeader) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  // Validate chunk_id format (defense-in-depth)
  if (!/^[a-f0-9]+$/i.test(chunkId)) {
    return NextResponse.json({ error: "Invalid chunk ID format" }, { status: 400 });
  }

  try {
    const response = await fetch(`${BACKEND_URL}/citations/${chunkId}`, {
      headers: {
        Authorization: authHeader,
      },
    });

    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    console.error("Citations proxy error:", error);
    return NextResponse.json({ error: "Failed to fetch citation" }, { status: 500 });
  }
}
