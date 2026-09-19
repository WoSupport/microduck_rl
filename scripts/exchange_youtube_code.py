#!/usr/bin/env python3
"""
Manually exchange authorization code or full redirected URL for YouTube token,
then immediately upload the YouTube Shorts video.
"""

import os
import sys
import json
import urllib.parse
from google_auth_oauthlib.flow import InstalledAppFlow

CONFIG_DIR = os.path.expanduser("~/.config/youtube")
os.makedirs(CONFIG_DIR, exist_ok=True)
SECRETS_FILE = os.path.join(CONFIG_DIR, "client_secrets.json")
STATE_FILE = os.path.join(CONFIG_DIR, "oauth_flow_state.json")
TOKEN_FILE = os.path.join(CONFIG_DIR, "youtube_token.json")
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

def main():
    if len(sys.argv) < 2:
        print("Usage: python exchange_youtube_code.py <CODE_OR_REDIRECT_URL>")
        sys.exit(1)

    raw_input = sys.argv[1].strip()
    code = raw_input

    # If full URL pasted, extract code parameter
    if "code=" in raw_input:
        parsed = urllib.parse.urlparse(raw_input)
        params = urllib.parse.parse_qs(parsed.query)
        code = params.get("code", [raw_input])[0]

    print(f"Exchanging authorization code: {code[:15]}...")

    if not os.path.exists(STATE_FILE):
        print(f"Error: {STATE_FILE} not found. Run youtube_oauth_server.py first to generate state.")
        sys.exit(1)

    with open(STATE_FILE, "r") as f:
        state_data = json.load(f)

    redirect_uri = state_data.get("redirect_uri", "http://localhost:8080/")
    code_verifier = state_data.get("code_verifier")

    flow = InstalledAppFlow.from_client_secrets_file(
        SECRETS_FILE,
        SCOPES,
        redirect_uri=redirect_uri
    )
    if code_verifier:
        flow.code_verifier = code_verifier

    flow.fetch_token(code=code)
    creds = flow.credentials

    with open(TOKEN_FILE, "w") as f:
        f.write(creds.to_json())

    print(f"✅ Successfully saved YouTube credentials to {TOKEN_FILE}!")
    print("\n🚀 Now uploading YouTube Short...")
    os.system("/home/ubuntu/vibeduck/microduck_rl/.venv/bin/python /home/ubuntu/vibeduck/scripts/upload_youtube_short.py")

if __name__ == "__main__":
    main()
