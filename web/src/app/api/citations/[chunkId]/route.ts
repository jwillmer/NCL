/**
 * Citations API Proxy Route
 *
 * Proxies citation requests to the Python backend.
 * Handles JWT validation and forwards the token for defense-in-depth auth.
 */

import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = (process.env.AGENT_URL || "http://localhost:8000/copilotkit").replace(
  "/copilotkit",
  ""
);

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ chunkId: string }> }
) {
  const { chunkId } = await params;

  // Extract token from Authorization header
  const authHeader = req.headers.get("authorization");
  if (!authHeader?.startsWith("Bearer ")) {
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
