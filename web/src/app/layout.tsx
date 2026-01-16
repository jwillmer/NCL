import type { Metadata } from "next";
import Script from "next/script";
import { AuthProvider } from "@/components/auth";
import "./globals.css";

export const metadata: Metadata = {
  title: "Maritime Technical Support System (MTSS)",
  description: "AI-powered technical support system for maritime crew - search issue history and knowledge base",
};

// In development, Next.js provides env vars via process.env
// In production (Docker), we load runtime config via /config.js
const isDev = process.env.NODE_ENV === "development";

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        {/* Runtime configuration for Docker deployments - only load in production */}
        {!isDev && <Script src="/config.js" strategy="beforeInteractive" />}
      </head>
      <body>
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
