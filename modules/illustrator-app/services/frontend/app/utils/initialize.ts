export async function initializeApp() {
  // Ping API to wake it up (if it's asleep) and check connectivity
  const { $api } = useNuxtApp();
  console.info("Pinging API to wake it up...");
  $api("/api/ping", { method: "GET" }).then(() => {
    console.info("API woke up successfully");
  }).catch((err) => {
    console.error("Error pinging API:", err);
  });

  // Startup workers
  await initWorkers();
}
