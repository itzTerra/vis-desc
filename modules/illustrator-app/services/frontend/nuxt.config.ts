import tailwindcss from "@tailwindcss/vite";

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
  devtools: { enabled: false },
  runtimeConfig: {
    public: {
      // can be overridden by NUXT_PUBLIC_* environment variables
      githubUrl: "https://github.com/itzTerra/vis-desc",
      appVersion: process.env.VITE_APP_VERSION || "1.0.0",
      commitHash: process.env.VITE_COMMIT_HASH || "unknown",
      buildTime: new Date().toISOString(),
      wsBaseUrl: process.env.VITE_API_WS_BASE_URL || "ws://localhost:8000",
      openFetch: {
        api: {
          baseURL: process.env.VITE_API_BASE_URL || "http://localhost:8000",
        },
      },
    },
    app: {
      baseURL: "/"
    }
  },
  vite: {
    plugins: [tailwindcss()],
    server: {
      proxy: {
        "/api": {
          target: process.env.VITE_API_BASE_URL || "http://localhost:8000",
          changeOrigin: true,
        },
      },
    },
    optimizeDeps: {
      include: [
        "lodash-es",
        "vue-pdf-embed"
      ]
    },
    build: {
      sourcemap: true // process.env.NODE_ENV === "development"
    }
  },
  css: ["~/assets/css/app.css"],
  modules: ["@vueuse/nuxt", "nuxt-open-fetch", "@nuxt/icon"],
  openFetch: {
    clients: {
      api: {
        baseURL: process.env.VITE_API_BASE_URL || "http://localhost:8000",
        schema: "http://api:8000/api/openapi.json"
      }
    }
  }
});
