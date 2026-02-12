import json
import base64
from django.test import TestCase, Client, override_settings
from unittest.mock import patch


class BatchImageEndpointsTest(TestCase):
    def setUp(self):
        self.client = Client()

    @patch("core.tools.text2image.generate_image_bytes")
    def test_gen_image_bytes_batch_success_ordering(self, mock_generate):
        # make generate return predictable bytes per prompt
        def gen(prompt):
            return f"IMG::{prompt}".encode("utf-8")

        mock_generate.side_effect = gen

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

    @patch("core.tools.llm.enhance_text_with_llm")
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
