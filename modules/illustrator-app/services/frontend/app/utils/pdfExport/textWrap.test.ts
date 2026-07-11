import assert from "node:assert/strict";
import test from "node:test";

import { wrapText } from "./textWrap.ts";

// Fixed-width fake measurer: 10 units per character, for predictable tests.
const measureWidth = (s: string) => s.length * 10;

test("wrapText keeps a short string on one line", () => {
  assert.deepEqual(wrapText("hello world", 200, measureWidth), ["hello world"]);
});

test("wrapText breaks onto a new line once the max width would be exceeded", () => {
  // "hello world" is 110 units; "hello" is 50, "hello world" is 110 > 100
  assert.deepEqual(wrapText("hello world", 100, measureWidth), ["hello", "world"]);
});

test("wrapText wraps multiple words across several lines", () => {
  const result = wrapText("one two three four five", 70, measureWidth);
  assert.deepEqual(result, ["one two", "three", "four", "five"]);
});

test("wrapText keeps an over-long single word on its own line rather than dropping it", () => {
  const result = wrapText("supercalifragilisticexpialidocious short", 100, measureWidth);
  assert.deepEqual(result, ["supercalifragilisticexpialidocious", "short"]);
});

test("wrapText collapses repeated whitespace", () => {
  assert.deepEqual(wrapText("hello    world", 200, measureWidth), ["hello world"]);
});

test("wrapText returns an empty array for empty input", () => {
  assert.deepEqual(wrapText("", 200, measureWidth), []);
});
