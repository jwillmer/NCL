/**
 * Archive API Proxy Route
 *
 * Proxies archive file requests to the Python backend.
 * Used for downloading original source files and viewing attachments.
 *
 * Supports two auth methods:
 * 1. Authorization header (for programmatic API calls)
 * 2. Supabase session cookie (for browser navigation/link clicks)
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
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path } = await params;
  const filePath = path.join("/");

  // Try Authorization header first
  let authHeader = req.headers.get("authorization");

  // Fall back to Supabase session cookie for browser navigation
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
      // Use getUser() which validates with Supabase server (more secure than getSession)
      const { data: { user }, error } = await supabase.auth.getUser();
      if (error) {
        console.error("Auth error:", error.message);
      }
      if (user) {
        // User is authenticated - get the session for the access token
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

  // Security: Reject paths with traversal attempts (defense-in-depth)
  if (filePath.includes("..") || filePath.startsWith("/") || filePath.startsWith("\\")) {
    return NextResponse.json({ error: "Invalid path" }, { status: 400 });
  }

  try {
    const response = await fetch(`${BACKEND_URL}/archive/${filePath}`, {
      headers: {
        Authorization: authHeader,
      },
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: response.statusText },
        { status: response.status }
      );
    }

    // Get the file content and content-type
    const content = await response.arrayBuffer();
    const contentType = response.headers.get("content-type") || "application/octet-stream";
    const contentDisposition = response.headers.get("content-disposition");

    const headers: HeadersInit = {
      "Content-Type": contentType,
    };

    if (contentDisposition) {
      headers["Content-Disposition"] = contentDisposition;
    }

    return new NextResponse(content, { headers });
  } catch (error) {
    console.error("Archive proxy error:", error);
    return NextResponse.json({ error: "Failed to fetch file" }, { status: 500 });
  }
}
