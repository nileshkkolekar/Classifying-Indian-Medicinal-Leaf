import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The built bundle is served by FastAPI itself, so the app always talks to
// its own origin and there is no CORS surface to configure or get wrong.
// During `npm run dev` Vite stands in for that by proxying the API paths.
const API = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
    // Sourcemaps would ship the whole app's source to anyone who opens
    // devtools on the deployed site.
    sourcemap: false,
  },
  server: {
    port: 5173,
    proxy: {
      "/auth": { target: API, changeOrigin: true },
      "/predict": { target: API, changeOrigin: true },
      "/jobs": { target: API, changeOrigin: true },
      "/health": { target: API, changeOrigin: true },
    },
  },
});
