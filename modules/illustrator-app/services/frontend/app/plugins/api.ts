// export default defineNuxtPlugin((nuxtApp) => {
//   const runtimeConfig = useRuntimeConfig();

//   const baseUrl = computed<string>(() => import.meta.server ? "http://api:8000" : runtimeConfig.public.baseApiUrl);

//   const api = $fetch.create({
//     baseURL: baseUrl.value,
//     headers: {
//       "Content-Type": "application/json",
//       // credentials: "include"
//     },
//     async onResponseError({ response }) {
//       if (response.status === 401) {
//         await nuxtApp.runWithContext(() => navigateTo("/login"));
//       }
//     }
//   });

//   // Expose to useNuxtApp().$api
//   return {
//     provide: {
//       baseUrl,
//       api
//     }
//   };
// });
