import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The production build is served by the CordiaAI FastAPI backend at /twin/,
// on the same origin as the main app, so it can reuse that app's login token
// to load a patient's record. In dev, /api is proxied to that backend.
// `npm run build:standalone` (mode "standalone") makes a self-contained copy
// with relative paths and no backend, for any static host: dist-standalone/.
// Five pages share one codebase: index.html (the clinical twin),
// digital.html (the holographic twin), workstation.html (the Cordia Heart
// Twin workstation with the 3D twin as its heart), cordia.html (the Cordia AI
// dashboard with the illustrated, labelled heart) and heart.html (that heart alone,
// in the reference look).
export default defineConfig(({ command, mode }) => ({
  plugins: [react()],
  base: mode === "standalone" ? "./" : command === "build" ? "/twin/" : "/",
  build: {
    ...(mode === "standalone" ? { outDir: "dist-standalone" } : {}),
    rollupOptions: { input: { main: "index.html", digital: "digital.html", workstation: "workstation.html", cordia: "cordia.html", heart: "heart.html" } },
  },
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
}));
