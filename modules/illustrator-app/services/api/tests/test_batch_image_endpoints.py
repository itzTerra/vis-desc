import json
import base64
from unittest.mock import patch

from django.test import TestCase, Client, override_settings


class BatchImageEndpointsTest(TestCase):
    def setUp(self):
        self.client = Client()

    @patch("core.tools.text2image.generate_image_bytes")
    def test_gen_image_bytes_batch_success_ordering(self, mock_generate):
        # make generate return predictable bytes per prompt
        mock_generate.side_effect = lambda prompt: f"IMG::{prompt}".encode("utf-8")

        texts = ["one", "two", "three"]
        resp = self.client.post(
            "/api/gen-image-bytes",
            data=json.dumps({"texts": texts}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.content)
        self.assertEqual(len(data), 3)
        for i, item in enumerate(data):
            self.assertTrue(item.get("ok"))
            img_b64 = item.get("image_b64")
            self.assertIsNotNone(img_b64)
            raw = base64.b64decode(img_b64).decode("utf-8")
            self.assertEqual(raw, f"IMG::{texts[i]}")

    @patch("core.api.enhance_text_with_llm")
    def test_enhance_batch_success_ordering(self, mock_enhance):
        mock_enhance.side_effect = lambda t: f"ENH::{t}"

        texts = ["a", "b", "c"]
        resp = self.client.post(
            "/api/enhance",
            data=json.dumps({"texts": texts}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.content)
        self.assertEqual(len(data), 3)
        for i, item in enumerate(data):
            self.assertTrue(item.get("ok"))
            self.assertEqual(item.get("text"), f"ENH::{texts[i]}")

    def test_gen_image_bytes_exceed_limit(self):
        texts = ["1", "2", "3"]
        # set max batch size to 2 to trigger validation
        with override_settings(IMAGE_GENERATION_MAX_BATCH_SIZE=2):
            resp = self.client.post(
                "/api/gen-image-bytes",
                data=json.dumps({"texts": texts}),
                content_type="application/json",
            )
            self.assertEqual(resp.status_code, 400)

    @patch("core.tools.text2image.generate_image_bytes")
    def test_gen_image_bytes_with_non_string_and_empty_item(self, mock_generate):
        # ensure generate_image_bytes is never called with non-strings, and
        # that an empty string is reported via the provider's ValueError
        def gen(prompt):
            if not isinstance(prompt, str):
                raise AssertionError("generate_image_bytes called with non-string")
            if not prompt.strip():
                raise ValueError("Prompt cannot be empty or whitespace")
            return f"IMG::{prompt}".encode("utf-8")

        mock_generate.side_effect = gen

        texts = ["valid", 123, ""]
        resp = self.client.post(
            "/api/gen-image-bytes",
            data=json.dumps({"texts": texts}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.content)
        self.assertEqual(len(data), 3)

        # first item succeeded
        self.assertTrue(data[0].get("ok"))

        # second item is non-string -> explicit error
        self.assertFalse(data[1].get("ok"))
        self.assertIn("text must be a string", data[1].get("error"))

        # third item is empty string -> provider-level ValueError propagated
        self.assertFalse(data[2].get("ok"))
        self.assertIn("Prompt cannot be empty", data[2].get("error"))
