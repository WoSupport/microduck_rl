#!/usr/bin/env python3
"""
OAuth 2.0 helper server for YouTube authentication.
1. Generates authorization URL with state & code verifier.
2. Listens on port 8080 for the OAuth redirect.
3. Automatically exchanges code for token, saves youtube_token.json,
   and uploads the YouTube Shorts video.
"""

import os
import sys
import json
import time
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

CONFIG_DIR = os.path.expanduser("~/.config/youtube")
os.makedirs(CONFIG_DIR, exist_ok=True)
SECRETS_FILE = os.path.join(CONFIG_DIR, "client_secrets.json")
STATE_FILE = os.path.join(CONFIG_DIR, "oauth_flow_state.json")
TOKEN_FILE = os.path.join(CONFIG_DIR, "youtube_token.json")
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
REDIRECT_URI = "http://localhost:8080/"

def init_flow():
    flow = InstalledAppFlow.from_client_secrets_file(
        SECRETS_FILE,
        SCOPES,
        redirect_uri=REDIRECT_URI
    )
    auth_url, state = flow.authorization_url(
        prompt="consent",
        access_type="offline"
    )
    
    # Save state so offline code exchange is possible if redirect misses
    state_data = {
        "auth_url": auth_url,
        "state": state,
        "code_verifier": flow.code_verifier,
        "redirect_uri": REDIRECT_URI
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state_data, f, indent=2)
        
    return flow, auth_url, state

flow_instance, auth_url_str, expected_state = init_flow()

class OAuthCallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        
        if "code" in params:
            code = params["code"][0]
            print(f"\n[OAuth Server] Received authorization code from browser: {code[:15]}...")
            
            try:
                flow_instance.fetch_token(code=code)
                creds = flow_instance.credentials
                with open(TOKEN_FILE, "w") as f:
                    f.write(creds.to_json())
                print(f"[OAuth Server] ✅ Successfully saved YouTube credentials to {TOKEN_FILE}")
                
                # HTML Success page
                self.send_response(200)
                self.send_header("Content-type", "text/html; charset=utf-8")
                self.end_headers()
                html = """
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="utf-8">
                    <title>YouTube Authentication Successful</title>
                    <style>
                        body { font-family: sans-serif; background: #0f172a; color: #f8fafc; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }
                        .card { background: #1e293b; padding: 40px; border-radius: 16px; text-align: center; max-width: 480px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }
                        h1 { color: #22c55e; margin-top: 0; }
                        p { color: #94a3b8; line-height: 1.6; }
                    </style>
                </head>
                <body>
                    <div class="card">
                        <h1>🦆 Auth Successful!</h1>
                        <p>Google has authorized the Microduck Uploader. You can close this browser tab now.</p>
                        <p style="color: #38bdf8; font-weight: bold;">Uploading your YouTube Short in the background...</p>
                    </div>
                </body>
                </html>
                """
                self.wfile.write(html.encode("utf-8"))
                
                # Trigger video upload in separate thread or exit
                import threading
                def trigger_upload():
                    time.sleep(1)
                    os.system("/home/ubuntu/vibeduck/microduck_rl/.venv/bin/python /home/ubuntu/vibeduck/scripts/upload_youtube_short.py")
                threading.Thread(target=trigger_upload, daemon=True).start()
                
            except Exception as e:
                print(f"[OAuth Server] ❌ Error exchanging code: {e}")
                self.send_response(500)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(f"OAuth exchange error: {e}".encode("utf-8"))
        else:
            self.send_response(400)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"No code parameter found in callback URL.")

    def log_message(self, format, *args):
        # Clean server logging
        pass

def main():
    print("\n" + "="*70)
    print("🔐 YOUTUBE OAUTH 2.0 SERVER READY")
    print("="*70)
    print(f"Authorization URL:\n{auth_url_str}\n")
    print("="*70)
    print("Starting callback listener on http://0.0.0.0:8080/ ...")
    server = HTTPServer(("0.0.0.0", 8080), OAuthCallbackHandler)
    server.serve_forever()

if __name__ == "__main__":
    main()
