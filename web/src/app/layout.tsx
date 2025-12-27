import type { Metadata } from "next";
import { AuthProvider } from "@/components/auth";
import "./globals.css";

export const metadata: Metadata = {
  title: "Maritime Technical Support System (MTSS)",
  description: "AI-powered technical support system for maritime crew - search issue history and knowledge base",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <AuthProvider>{children}</AuthProvider>
      </body>
    </html>
  );
}
