import VuePdfEmbed from "vue-pdf-embed";

export default defineNuxtPlugin({
  parallel: true,
  async setup(nuxtApp) {
    nuxtApp.vueApp.component("VuePdfEmbed", VuePdfEmbed);
  }
});
