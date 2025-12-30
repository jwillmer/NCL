/**
 * Archive API Proxy Route
 *
 * Proxies archive file requests to the Python backend.
 * Used for downloading original source files.
 */

import { NextRequest, NextResponse } from "next/server";

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

  // Extract token from Authorization header
  const authHeader = req.headers.get("authorization");
  if (!authHeader?.startsWith("Bearer ")) {
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
