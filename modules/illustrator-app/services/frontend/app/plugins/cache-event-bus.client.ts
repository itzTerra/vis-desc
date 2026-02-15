type CacheEventMap = {
  "cache:model-needed": { groupId: string }
}

type EventHandler<T = any> = (payload: T) => void

class SimpleEventBus {
  private listeners: Map<string, Set<EventHandler>> = new Map();

  on<K extends keyof CacheEventMap>(event: K, handler: EventHandler<CacheEventMap[K]>): () => void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(handler);

    return () => {
      this.listeners.get(event)?.delete(handler);
    };
  }

  off<K extends keyof CacheEventMap>(event: K, handler: EventHandler<CacheEventMap[K]>): void {
    this.listeners.get(event)?.delete(handler);
  }

  emit<K extends keyof CacheEventMap>(event: K, payload: CacheEventMap[K]): void {
    this.listeners.get(event)?.forEach((handler) => {
      handler(payload);
    });
  }
}

let eventBusInstance: SimpleEventBus | null = null;

function getEventBusInstance(): SimpleEventBus {
  if (!eventBusInstance) {
    eventBusInstance = new SimpleEventBus();
  }
  return eventBusInstance;
}

export const useCacheEvents = () => {
  const bus = getEventBusInstance();

  return {
    on: bus.on.bind(bus),
    off: bus.off.bind(bus),
    emit: bus.emit.bind(bus),
  };
};

export default defineNuxtPlugin(() => {
  const bus = getEventBusInstance();

  return {
    provide: {
      cacheEvents: {
        on: bus.on.bind(bus),
        off: bus.off.bind(bus),
        emit: bus.emit.bind(bus),
      },
    },
  };
});
