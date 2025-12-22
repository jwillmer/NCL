# NCL Frontend

Web interface for the NCL Email Archive system, built with React, TypeScript, and CopilotKit.

## Tech Stack

- **Framework:** React 18 + TypeScript
- **Build Tool:** Vite
- **Styling:** TailwindCSS
- **UI Components:** Radix UI
- **AI Chat:** CopilotKit
- **Authentication:** Supabase Auth
- **Testing:** Vitest + Testing Library + MSW

## Prerequisites

- Node.js 18+
- npm or yarn
- Running NCL API backend (see parent README)
- Supabase project with authentication enabled

## Setup

1. Install dependencies:

```bash
cd frontend
npm install
```

2. Create environment file:

```bash
cp .env.local.example .env.local
```

3. Configure environment variables in `.env.local`:

```
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key
VITE_API_URL=http://localhost:8000
```

## Development

Start the development server:

```bash
npm run dev
```

The frontend will be available at http://localhost:5173

## Building for Production

```bash
npm run build
```

The build output will be in the `dist/` directory.

## Testing

```bash
# Run tests
npm run test

# Run tests with UI
npm run test:ui
```

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── auth.tsx           # Auth provider, login form, user menu
│   │   ├── ChatContainer.tsx  # CopilotKit chat interface
│   │   ├── Layout.tsx         # Header and main layout
│   │   ├── ProtectedRoute.tsx # Auth route guard
│   │   ├── Sources.tsx        # Source display components
│   │   └── ui.tsx             # Base UI components (Radix)
│   ├── lib/
│   │   ├── supabase.ts        # Supabase client
│   │   └── utils.ts           # Utility functions
│   ├── styles/
│   │   └── globals.css        # Global styles + Tailwind
│   ├── test/
│   │   ├── mocks/             # MSW mock handlers
│   │   └── setup.ts           # Test setup
│   ├── types/
│   │   └── sources.ts         # TypeScript types
│   ├── App.tsx                # Root component
│   └── main.tsx               # Entry point
├── public/
│   └── ncl-logo.svg           # Logo
└── package.json
```

## Authentication

The app supports multiple authentication methods via Supabase:

- Email/Password
- Magic Link (passwordless)
- OAuth providers (Google, GitHub, Azure)

Configure OAuth providers in your Supabase dashboard.

## NCL Brand Colors

| Color | Hex | Usage |
|-------|-----|-------|
| NCL Blue | `#003A8F` | Primary, headers, buttons |
| Dark Navy | `#001F5B` | Secondary, hover states |
| Light Blue | `#4F83CC` | Accents |
| Gray | `#6D6E71` | Secondary text |
| Light Gray | `#D1D3D4` | Borders, backgrounds |
