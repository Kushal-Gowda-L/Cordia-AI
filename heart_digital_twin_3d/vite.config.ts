import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The production build is served by the CordiaAI FastAPI backend at /twin/,
// on the same origin as the main app, so it can reuse that app's login token
// to load a patient's record. In dev, /api is proxied to that backend.
// `npm run build:standalone` (mode "standalone") makes a self-contained copy
// with relative paths and no backend, for any static host: dist-standalone/.
export default defineConfig(({ command, mode }) => ({
  plugins: [react()],
  base: mode === "standalone" ? "./" : command === "build" ? "/twin/" : "/",
  build: mode === "standalone" ? { outDir: "dist-standalone" } : {},
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
}));
