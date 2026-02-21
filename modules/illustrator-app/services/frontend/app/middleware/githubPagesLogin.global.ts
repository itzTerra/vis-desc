import { defineNuxtRouteMiddleware, useRuntimeConfig, navigateTo } from 'nuxt/app'

export default defineNuxtRouteMiddleware((to, from) => {
  if (typeof window !== 'undefined' && window.location.hostname.endsWith('github.io')) {
    const config = useRuntimeConfig()
    const password = config.public.loginPassword

    const inputPassword = window.prompt('Enter password:')

    if (inputPassword !== password) {
      window.alert('Invalid password.')
      return abortNavigation();
    }
  }
})
