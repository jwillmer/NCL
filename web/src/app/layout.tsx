import type { Metadata } from "next";
import { AuthProvider } from "@/components/auth";
import "./globals.css";

export const metadata: Metadata = {
  title: "NCL Email Archive",
  description: "Intelligent Document Search powered by AI",
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
