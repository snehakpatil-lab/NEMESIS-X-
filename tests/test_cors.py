import unittest

from fastapi.testclient import TestClient

from app.service import app


class CorsTests(unittest.TestCase):
    def test_production_frontend_can_call_api_and_preflight_post(self):
        origin = "https://nemesisx2026.vercel.app"
        with TestClient(app) as client:
            response = client.get("/openapi.json", headers={"Origin": origin})
            preflight = client.options(
                "/api/videos/1/analyze",
                headers={
                    "Origin": origin,
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type",
                },
            )

        self.assertEqual(
            response.headers.get("access-control-allow-origin"),
            origin,
        )
        self.assertEqual(preflight.status_code, 200)
        self.assertEqual(
            preflight.headers.get("access-control-allow-origin"),
            origin,
        )
        self.assertIn("POST", preflight.headers.get("access-control-allow-methods", ""))
