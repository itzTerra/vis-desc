import assert from "node:assert/strict";
import test from "node:test";

import { dataUriToBytes } from "./imageReencode.ts";

test("dataUriToBytes decodes the base64 payload back to the original bytes", () => {
  const original = new Uint8Array([1, 2, 3, 250, 0, 255]);
  const base64 = Buffer.from(original).toString("base64");
  const dataUri = `data:application/octet-stream;base64,${base64}`;

  assert.deepEqual(Array.from(dataUriToBytes(dataUri)), Array.from(original));
});

test("dataUriToBytes handles a real-shaped image/jpeg data URI prefix", () => {
  const original = new Uint8Array([0xff, 0xd8, 0xff, 0xe0]); // JPEG magic bytes
  const base64 = Buffer.from(original).toString("base64");
  const dataUri = `data:image/jpeg;base64,${base64}`;

  assert.deepEqual(Array.from(dataUriToBytes(dataUri)), Array.from(original));
});
