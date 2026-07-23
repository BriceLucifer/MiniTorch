import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "node:path";

export default defineConfig({
  plugins: [react()],
  define: {
    "process.env": {}
  },
  build: {
    outDir: "../../MiniTorch/visualization/static",
    emptyOutDir: true,
    cssCodeSplit: false,
    lib: {
      entry: resolve(__dirname, "src/main.tsx"),
      name: "MiniTorchModelViewer",
      formats: ["iife"],
      fileName: () => "model-viewer.js"
    },
    rollupOptions: {
      output: {
        assetFileNames: (assetInfo) =>
          assetInfo.name?.endsWith(".css")
            ? "model-viewer.css"
            : "assets/[name][extname]",
        inlineDynamicImports: true
      }
    }
  }
});
