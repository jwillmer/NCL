import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";
import path from "path";

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: "autoUpdate",
      injectRegister: "auto",
      devOptions: {
        enabled: true,
        type: "module",
        navigateFallback: "/index.html",
      },
      manifest: {
        name: "MTSS",
        short_name: "MTSS",
        description: "Vessel issue history & knowledge base",
        theme_color: "#0b2545",
        background_color: "#ffffff",
        display: "standalone",
        start_url: "/",
        scope: "/",
        icons: [
          { src: "/icons/mtss-192.png", sizes: "192x192", type: "image/png" },
          { src: "/icons/mtss-512.png", sizes: "512x512", type: "image/png" },
          { src: "/icons/mtss-maskable-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
        ],
      },
      workbox: {
        globPatterns: ["**/*.{js,css,html,svg,woff2,png,ico}"],
        navigateFallback: "/index.html",
        navigateFallbackDenylist: [/^\/api/, /^\/config\.js/, /^\/health/, /^\/docs/, /^\/redoc/],
        // NOTE: /config.js is intentionally NOT runtime-cached. It's ~500
        // bytes, served with `no-cache, no-store, must-revalidate`, and
        // carries the Supabase/API env values the SPA bootstraps from —
        // caching it risks one broken deploy poisoning every user's PWA
        // with a stale or empty config. The navigateFallbackDenylist above
        // keeps the SW from returning index.html for this URL.
        runtimeCaching: [
          {
            urlPattern: ({ url }) =>
              url.pathname.startsWith("/api/vessels") ||
              url.pathname.startsWith("/api/vessel-types") ||
              url.pathname.startsWith("/api/vessel-classes"),
            handler: "StaleWhileRevalidate",
            options: {
              cacheName: "vessels",
              expiration: { maxAgeSeconds: 3600, maxEntries: 10 },
              cacheableResponse: { statuses: [200] },
            },
          },
          {
            urlPattern: ({ url }) => url.pathname === "/api/stats",
            handler: "StaleWhileRevalidate",
            options: {
              cacheName: "corpus-stats",
              expiration: { maxAgeSeconds: 600, maxEntries: 1 },
              cacheableResponse: { statuses: [200] },
            },
          },
          {
            urlPattern: ({ url }) => url.pathname.startsWith("/api/archive/"),
            handler: "CacheFirst",
            options: {
              cacheName: "archive-files",
              expiration: { maxAgeSeconds: 60 * 60 * 24 * 30, maxEntries: 500 },
              cacheableResponse: { statuses: [200] },
            },
          },
        ],
      },
    }),
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/config.js": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/health": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
    headers: {
      // Mirror FastAPI SecurityHeadersMiddleware for dev parity
      "X-Content-Type-Options": "nosniff",
      "X-Frame-Options": "DENY",
      "X-XSS-Protection": "1; mode=block",
      "Referrer-Policy": "strict-origin-when-cross-origin",
    },
  },
  build: {
    outDir: "dist",
    rollupOptions: {
      output: {
        manualChunks(id: string) {
          if (!id.includes("node_modules")) {
            return undefined;
          }
          if (id.includes("@radix-ui/")) {
            return "radix";
          }
          if (
            id.includes("react-markdown") ||
            id.includes("remark-") ||
            id.includes("rehype-")
          ) {
            return "markdown";
          }
          if (id.includes("@ai-sdk/")) {
            return "ai-sdk";
          }
          // Match the top-level `ai` package but not paths like `chai` or `ai-sdk`.
          if (/[\\/]node_modules[\\/]ai[\\/]/.test(id)) {
            return "ai-sdk";
          }
          return undefined;
        },
      },
    },
  },
});
