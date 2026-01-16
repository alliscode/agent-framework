import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

// https://vite.dev/config/
export default defineConfig({
  base: "",
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    commonjsOptions: {
      // Enable deterministic builds, as per https://github.com/vitejs/vite/issues/13672#issuecomment-1784110536
      strictRequires: true,
    },
    outDir: "../agent_framework_devui/ui",
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks: undefined,
        inlineDynamicImports: true,
        // Use content hashes for cache busting
        entryFileNames: "assets/index-[hash].js",
        chunkFileNames: "assets/[name]-[hash].js",
        assetFileNames: "assets/[name]-[hash].[ext]",
      },
    },
  },
});
