import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// Minimal Vite + React UI for the OwnedLM repo status page.
// The previous TanStack Start / Nitro scaffold referenced missing scripts and routes.
export default defineConfig({
  server: {
    host: "0.0.0.0",
    port: 8080,
  },
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
});
