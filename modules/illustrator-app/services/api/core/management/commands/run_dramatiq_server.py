# management/commands/run_worker_server.py
from django.core.management.base import BaseCommand
from django.core.management import call_command
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import os


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Worker alive")


class Command(BaseCommand):
    def handle(self, *args, **options):
        port = int(os.environ.get("PORT", 8000))

        # Start health server
        server = HTTPServer(("0.0.0.0", port), HealthHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        # Start Dramatiq worker
        call_command("rundramatiq", "-p", "2", "-t", "1")
