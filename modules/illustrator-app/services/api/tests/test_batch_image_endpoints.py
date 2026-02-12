import base64
import json
from unittest.mock import patch

from django.test import TestCase, override_settings


class BatchImageEndpointsTest(TestCase):
    def post(self, path, payload):
        return self.client.post(
            path, data=json.dumps(payload), content_type="application/json"
        )

    def json_from(self, resp):
        return json.loads(resp.content)

    def assert_ok_true(self, item):
        self.assertIn("ok", item)
        self.assertIs(item["ok"], True)

    def decode_image(self, item):
        img_b64 = item["image_b64"]
        self.assertIsInstance(img_b64, str)
        return base64.b64decode(img_b64).decode("utf-8")

    @patch("core.tools.text2image.generate_image_bytes")
    def test_gen_image_bytes_batch_success_ordering(self, mock_generate):
        mock_generate.side_effect = lambda prompt: f"IMG::{prompt}".encode("utf-8")

        texts = ["one", "two", "three"]
        resp = self.post("/api/gen-image-bytes", {"texts": texts})
        self.assertEqual(resp.status_code, 200)
        data = self.json_from(resp)
        self.assertEqual(len(data), 3)
        for i, item in enumerate(data):
            self.assert_ok_true(item)
            raw = self.decode_image(item)
            self.assertEqual(raw, f"IMG::{texts[i]}")

    @patch("core.api.enhance_text_with_llm")
    def test_enhance_batch_success_ordering(self, mock_enhance):
        mock_enhance.side_effect = lambda t: f"ENH::{t}"

        texts = ["a", "b", "c"]
        resp = self.post("/api/enhance", {"texts": texts})
        self.assertEqual(resp.status_code, 200)
        data = self.json_from(resp)
        self.assertEqual(len(data), 3)
        for i, item in enumerate(data):
            self.assert_ok_true(item)
            self.assertIn("text", item)
            self.assertIsInstance(item["text"], str)
            self.assertEqual(item["text"], f"ENH::{texts[i]}")

    def test_gen_image_bytes_exceed_limit(self):
        texts = ["1", "2", "3"]
        with override_settings(IMAGE_GENERATION_MAX_BATCH_SIZE=2):
            resp = self.post("/api/gen-image-bytes", {"texts": texts})
            self.assertEqual(resp.status_code, 400)

    @patch("core.tools.text2image.generate_image_bytes")
    def test_gen_image_bytes_with_non_string_and_empty_item(self, mock_generate):
        def gen(prompt):
            if not isinstance(prompt, str):
                raise AssertionError("generate_image_bytes called with non-string")
            if not prompt.strip():
                raise ValueError("Prompt cannot be empty or whitespace")
            return f"IMG::{prompt}".encode("utf-8")

        mock_generate.side_effect = gen

        texts = ["valid", 123, ""]
        resp = self.post("/api/gen-image-bytes", {"texts": texts})
        self.assertEqual(resp.status_code, 200)
        data = self.json_from(resp)
        self.assertEqual(len(data), 3)

        self.assert_ok_true(data[0])

        self.assertIn("ok", data[1])
        self.assertIs(data[1]["ok"], False)
        self.assertIn("error", data[1])
        self.assertIsInstance(data[1]["error"], str)
        self.assertIn("string", data[1]["error"].lower())

        self.assertIn("ok", data[2])
        self.assertIs(data[2]["ok"], False)
        self.assertIn("error", data[2])
        self.assertIsInstance(data[2]["error"], str)
        self.assertIn("empty", data[2]["error"].lower())
