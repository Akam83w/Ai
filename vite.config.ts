import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { tanstackStart } from "@tanstack/react-start/plugin/vite";
import { nitro } from "nitro/vite";
import { appEnvPlugin } from "./scripts/app-env-plugin.mjs";
import path from "node:path";

export default defineConfig({
  server: {
    host: "0.0.0.0",
    port: 8080,
    allowedHosts: true,
  },
  plugins: [
    appEnvPlugin(),
    tanstackStart({
      srcDirectory: "src",
      router: {
        entry: "./router.tsx",
      },
    }),
    react(),
    tailwindcss(),
    nitro({
      preset: "node-server",
      compatibilityDate: "2025-01-01",
    }),
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
});
