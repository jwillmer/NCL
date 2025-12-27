# Authentication Flow

NCL uses Supabase Auth with JWT tokens validated in the Next.js API route. This document explains how authentication works across the 2-tier architecture.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Next.js App (port 3000)                 │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  React Frontend                                         ││
│  │  - Supabase Auth (login/logout)                         ││
│  │  - CopilotKit with Authorization header                 ││
│  └─────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────┐│
│  │  /api/copilotkit (API Route)                            ││
│  │  - JWT validation via Supabase getUser()                ││
│  │  - Forwards authenticated requests to Python agent      ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Python Agent (FastAPI, port 8000)              │
│  - Trusts requests from Next.js API route                   │
└─────────────────────────────────────────────────────────────┘
```

**Key Design Decision:** Authentication is enforced in the Next.js API route, not the Python agent. The API route acts as a secure gateway - if a request reaches the Python agent, it has already been authenticated.

## Authentication Flow

### 1. User Login (Frontend)

```typescript
// web/src/components/auth.tsx
<Auth
  supabaseClient={supabase}
  providers={["google", "github", "azure"]}
  magicLink={true}
/>
```

- User logs in via Supabase Auth UI (OAuth or magic link)
- Supabase returns a session with JWT access token
- Frontend stores session in local storage

### 2. Token Attachment (Frontend)

```typescript
// web/src/app/page.tsx
<CopilotKit
  runtimeUrl="/api/copilotkit"
  headers={{
    Authorization: `Bearer ${session.access_token}`,
  }}
>
```

- Every CopilotKit request includes the `Authorization: Bearer <token>` header
- Token is the Supabase access token (JWT)

### 3. Token Validation (API Route)

```typescript
// web/src/app/api/copilotkit/route.ts
async function verifyAuth(req: NextRequest): Promise<boolean> {
  const authHeader = req.headers.get("authorization");
  if (!authHeader?.startsWith("Bearer ")) {
    return false;
  }

  const token = authHeader.slice(7);
  const supabase = createClient(supabaseUrl, supabaseKey);
  const { data: { user }, error } = await supabase.auth.getUser(token);

  return !error && !!user;
}

export const POST = async (req: NextRequest) => {
  if (!(await verifyAuth(req))) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }
  // Forward to Python agent...
};
```

The API route validates tokens using Supabase's `getUser()` method, which verifies the JWT server-side.

### 4. Request Forwarding (API Route → Agent)

After successful authentication, the API route forwards the request to the Python agent via CopilotKit's `HttpAgent`. The Python agent trusts all requests from the API route.

## Configuration

### Next.js App (`web/.env.local`)

```bash
# Supabase configuration (required)
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key

# Python agent URL (must include /copilotkit path)
AGENT_URL=http://localhost:8000/copilotkit
```

### Python Agent (`.env`)

```bash
# CORS configuration
CORS_ORIGINS=http://localhost:3000
API_HOST=0.0.0.0
API_PORT=8000
```

## Development vs Production

| Setting | Development | Production |
|---------|-------------|------------|
| Auth validation | Always enabled | Always enabled |
| `CORS_ORIGINS` | localhost:3000 | Your domain(s) |
| `AGENT_URL` | localhost:8000 | Internal/private URL |

## Error Handling

### 401 Unauthorized

Returned when:
- No `Authorization` header present
- Token format invalid (not `Bearer <token>`)
- Supabase `getUser()` fails (expired, invalid token)

### Logging

Auth errors are logged to the console:
```
Auth error: Token expired
Missing Supabase configuration
```

## Security Considerations

1. **Anon Key:** The `NEXT_PUBLIC_SUPABASE_ANON_KEY` is safe for client-side use - access is controlled by Row Level Security
2. **CORS:** Set `CORS_ORIGINS` in the Python agent to your production domain(s)
3. **HTTPS:** Always use HTTPS in production
4. **Token Expiry:** Supabase tokens expire after 1 hour; frontend handles refresh automatically
5. **Python Agent:** Does not validate tokens - relies on API route as gateway

## Troubleshooting

### "Unauthorized" response
- Ensure `session.access_token` is being passed to CopilotKit headers
- Check if user is logged in before rendering CopilotKit
- Token may have expired; check auth state

### Auth working locally but not in production
- Verify `CORS_ORIGINS` includes your production domain
- Check all environment variables are set in production
- Ensure the Python agent is accessible from the Next.js server
