import * as protobuf from "protobufjs";
import { SPACY_PROTO } from "~/utils/spacy_proto";

type Req = {
  id: string;
  action: "fetchJson" | "fetchProto";
  url: string;
  texts: string[];
};

type Resp = { id: string; ok: true; contexts: any[] } | { id: string; ok: false; error: string };

let SpaCyContextsType: any | null = null;

const initProto = () => {
  if (SpaCyContextsType) return;
  const parsed = protobuf.parse(SPACY_PROTO);
  const root = parsed.root;
  SpaCyContextsType = root.lookupType("spacy.SpaCyContexts");
};

self.addEventListener("message", async (ev: MessageEvent<Req>) => {
  const msg = ev.data;
  try {
    if (msg.action === "fetchJson") {
      const res = await fetch(msg.url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texts: msg.texts }),
      });
      if (!res.ok) {
        const txt = await res.text();
        const err: Resp = { id: msg.id, ok: false, error: `fetch failed ${res.status} ${txt}` };
        self.postMessage(err);
        return;
      }
      const json = await res.json();
      const contexts = (json.contexts || []);
      const out: Resp = { id: msg.id, ok: true, contexts };
      self.postMessage(out);
      return;
    }

    if (msg.action === "fetchProto") {
      initProto();
      const res = await fetch(msg.url, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/x-protobuf" },
        body: JSON.stringify({ texts: msg.texts }),
      });
      if (!res.ok) {
        const txt = await res.text();
        const err: Resp = { id: msg.id, ok: false, error: `fetch failed ${res.status} ${txt}` };
        self.postMessage(err);
        return;
      }
      const buf = await res.arrayBuffer();
      const u8 = new Uint8Array(buf);
      const msgProto = (SpaCyContextsType as any).decode(u8);
      const obj = (SpaCyContextsType as any).toObject(msgProto, { longs: String, enums: String, defaults: true });
      const contexts = (obj.contexts || []).map((c: any) => ({
        tokens: c.tokens || [],
        sentences: (c.sentences || []).map((s: any) => ({ root: s.root || null, start: s.start || 0, end: s.end || 0 })),
        nounChunks: (c.noun_chunks || []).map((n: any) => ({ start: n.start || 0, end: n.end || 0, text: n.text || "" })),
      }));
      const out: Resp = { id: msg.id, ok: true, contexts };
      self.postMessage(out);
      return;
    }
  } catch (e: any) {
    const err: Resp = { id: msg.id, ok: false, error: String(e?.message ?? e) };
    self.postMessage(err);
  }
});

export {};
