import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "node:path";

const normalizeGeneratedWhitespace = {
  name: "normalize-generated-whitespace",
  generateBundle(
    _options: unknown,
    bundle: Record<string, { type: string; code?: string; source?: string }>
  ) {
    for (const output of Object.values(bundle)) {
      if (output.type === "chunk" && output.code !== undefined) {
        output.code = output.code.replace(/[ \t]+$/gm, "");
      } else if (typeof output.source === "string") {
        output.source = output.source.replace(/[ \t]+$/gm, "");
      }
    }
  }
};

export default defineConfig({
  plugins: [react(), normalizeGeneratedWhitespace],
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
