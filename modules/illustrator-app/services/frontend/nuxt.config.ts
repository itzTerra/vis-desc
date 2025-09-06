import tailwindcss from "@tailwindcss/vite";
import { readFileSync } from "fs";
import { execSync } from "child_process";

const packageJson = JSON.parse(readFileSync("package.json", "utf8"));
let gitHash = "unknown";

try {
  gitHash = execSync("git rev-parse --short HEAD", { encoding: "utf8" }).trim();
} catch {
  console.warn("Could not get git hash");
}

export default defineNuxtConfig({
  app: {
    head: {
      title: "vd-Illustrator",
      htmlAttrs: {
        lang: "en",
      },
      link: [
        { rel: "icon", type: "image/x-icon", href: "/favicon.ico" },
      ]
    }
  },
  ssr: false,
  compatibilityDate: "2024-11-01",
  devtools: { enabled: true },
  runtimeConfig: {
    public: {
      // can be overridden by NUXT_PUBLIC_BASE_*_URL environment variables
      baseApiUrl: "",
      baseUrl: "",
      githubUrl: "https://github.com/itzTerra/vis-desc",
      appVersion: packageJson.version,
      gitHash,
      buildTime: new Date().toISOString()
    }
  },
  vite: {
    plugins: [tailwindcss()],
    server: {
      proxy: {
        "/api": {
          target: process.env.VITE_API_BASE_URL,
          changeOrigin: true,
        },
      },
    },
    optimizeDeps: {
      include: [
        "lodash-es",
        "vue-pdf-embed"
      ]
    }
  },
  css: ["~/assets/css/app.css", "~/assets/css/main.css"],
  modules: ["@vueuse/nuxt", "nuxt-open-fetch", "@nuxt/icon"],
  openFetch: {
    clients: {
      api: {
        baseURL: process.env.VITE_API_BASE_URL,
        schema: "http://api:8000/api/openapi.json"
      }
    }
  }
});
