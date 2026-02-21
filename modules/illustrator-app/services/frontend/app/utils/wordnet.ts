import { loadAsync } from "jszip";

export type WordNetData = Record<string, Record<string, any>>;

let cacheName = "wordnet-cache";
const inMemoryWordnets = new Map<string, WordNetData>();

export async function isCached(idOrUrl: string): Promise<boolean> {
  try {
    const cache = await caches.open(cacheName);
    const keys = await cache.keys();
    return keys.some((r) => r.url.includes(idOrUrl));
  } catch {
    return false;
  }
}

async function loadFromCache(idOrUrl: string): Promise<WordNetData | null> {
  try {
    const cache = await caches.open(cacheName);
    const keys = await cache.keys();
    const found = keys.find((r) => r.url.includes(idOrUrl));
    if (!found) return null;

    const resp = await cache.match(found);
    if (!resp) return null;

    const json = await resp.json();
    return json as WordNetData;
  } catch {
    return null;
  }
}

export async function downloadAndLoadWordNet(id: string, url: string, customCacheName?: string, onProgress?: (p: number) => void): Promise<WordNetData> {
  if (customCacheName) cacheName = customCacheName;

  // If already in memory return
  if (inMemoryWordnets.has(id)) return inMemoryWordnets.get(id)!;

  // Try cache first
  const cached = await loadFromCache(id) || await loadFromCache(url);
  if (cached) {
    inMemoryWordnets.set(id, cached);
    return cached;
  }

  // Fetch resource. Support zipped JSON archives (.zip) by extracting first .json entry.
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to download WordNet: ${response.statusText}`);

  onProgress?.(10);

  const contentType = response.headers.get("content-type") || "";
  let parsed: any = null;

  if (url.endsWith(".zip") || contentType.includes("zip")) {
    // load arrayBuffer, use JSZip from CDN to unzip
    const buffer = await response.arrayBuffer();
    onProgress?.(30);

    const zip = await loadAsync(buffer);
    onProgress?.(50);

    // merge all .json files in the archive into a single object
    const merged: Record<string, any> = {};
    const fileNames = Object.keys(zip.files).filter((n) => n.toLowerCase().endsWith(".json"));
    if (fileNames.length === 0) throw new Error("No JSON file found in WordNet zip archive");

    for (const name of fileNames) {
      const txt = await zip.file(name)?.async("string");
      if (!txt) continue;
      try {
        const obj = JSON.parse(txt);
        for (const [k, v] of Object.entries(obj)) {
          const key = (k as string);
          const lkey = key.toLowerCase();
          if (!merged[lkey]) merged[lkey] = {};
          const entry = merged[lkey];
          for (const posKey of Object.keys(v as any)) {
            entry[posKey] = (entry[posKey] || { pronunciation: [], sense: [] });
            const src = (v as any)[posKey];
            if (src && typeof src === "object") {
              if (Array.isArray(src.sense)) {
                entry[posKey].sense = entry[posKey].sense.concat(src.sense);
              }
              if (Array.isArray(src.pronunciation)) {
                entry[posKey].pronunciation = entry[posKey].pronunciation.concat(src.pronunciation);
              }
            }
          }
        }
      } catch (err) {
        console.warn("Skipping invalid JSON in zip entry", name, err);
      }
    }

    parsed = merged;
  } else {
    // not a zip — assume JSON (object mapping lemmas)
    parsed = await response.json();
    // normalize keys to lowercase
    const normalized: Record<string, any> = {};
    for (const [k, v] of Object.entries(parsed)) {
      normalized[k.toLowerCase()] = v;
    }
    parsed = normalized;
  }

  onProgress?.(90);

  // attempt to cache the original response body where feasible
  try {
    const cache = await caches.open(cacheName);
    // store original resource request (some browsers disallow caching cross-origin opaque responses)
    try {
      await cache.put(new Request(`${url}#${id}`), new Response(JSON.stringify(parsed), { headers: { "Content-Type": "application/json" } }));
    } catch {
      // fallback: try caching the original response clone if available
      try {
        const rclone = response.clone();
        await cache.put(new Request(`${url}#${id}`), rclone);
      } catch (err) {
        console.warn("Failed to cache WordNet resource (zip/json)", err);
      }
    }
  } catch (err) {
    console.warn("Failed to open WordNet cache", err);
  }

  onProgress?.(100);

  inMemoryWordnets.set(id, parsed as WordNetData);
  return parsed as WordNetData;
}

export function synsets(word: string, pos?: string, id: string = "oewn"): any[] {
  const data = inMemoryWordnets.get(id);
  if (!data) {
    throw new Error(`WordNet with id "${id}" not loaded in memory. Call downloadAndLoadWordNet first.`);
  }

  const lemma = word.toLowerCase();
  const entry = data[lemma] || data[word] || data[word.toLowerCase()];
  if (!entry) return [];

  if (!pos) {
    const all: any[] = [];
    for (const p of Object.keys(entry)) {
      if (entry[p] && Array.isArray(entry[p].sense)) all.push(...entry[p].sense);
    }
    return all;
  }

  const posEntry = entry[pos];
  if (!posEntry || !Array.isArray(posEntry.sense)) return [];
  return posEntry.sense;
}

export function getPolysemyCount(word: string, id: string = "oewn", pos?: string): number {
  const senses = synsets(word, pos, id);
  return senses.length;
}

export const UD_TO_WN_POS: Record<string, string> = {
  NOUN: "n",
  VERB: "v",
  ADJ: "a",
  ADV: "r",
};

export function createWordNetContext(id: string = "oewn") {
  return {
    synsets: (word: string, options?: { pos?: string }) => {
      if (options && options.pos) {
        return synsets(word, options.pos, id);
      }
      return synsets(word, undefined, id);
    },
  };
}

export function clearWordNetFromMemory(id: string) {
  inMemoryWordnets.delete(id);
}

export async function clearWordNetCache(idOrUrl: string) {
  try {
    const cache = await caches.open(cacheName);
    const keys = await cache.keys();
    const matches = keys.filter((r) => r.url.includes(idOrUrl));
    await Promise.all(matches.map((k) => cache.delete(k)));
  } catch (err) {
    console.warn("Failed to clear WordNet cache", err);
  }
}

export type WordNetInfo = {
  id: string;
  downloadUrl: string;
};

export const WORDNETS: WordNetInfo[] = [
  {
    id: "oewn",
    downloadUrl: "/english-wordnet-2025-json.zip",
  },
];
