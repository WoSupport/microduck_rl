#!/usr/bin/env python3
"""
Upload Microduck Tail Wag YouTube Short via YouTube Data API v3.
Supports headless authentication with local server or redirect URL paste.
"""

import os
import sys
import json
import argparse
from pathlib import Path

def get_authenticated_service(secrets_file="client_secrets.json", token_file="youtube_token.json", port=8080):
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request

    SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
    creds = None

    if os.path.exists(token_file):
        try:
            creds = Credentials.from_authorized_user_file(token_file, SCOPES)
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
                with open(token_file, "w") as f:
                    f.write(creds.to_json())
        except Exception as e:
            print(f"Warning: cached token invalid ({e}). Re-authenticating...")
            creds = None

    if not creds or not creds.valid:
        if not os.path.exists(secrets_file):
            print(f"\n❌ Error: '{secrets_file}' not found!")
            print("To upload via YouTube API, download OAuth 2.0 Client Credentials JSON from Google Cloud Console")
            print(f"and save it as '{secrets_file}'.")
            sys.exit(1)

        flow = InstalledAppFlow.from_client_secrets_file(secrets_file, SCOPES)
        
        # In a headless environment, generate URL
        print("\n" + "="*60)
        print("🔐 GOOGLE / YOUTUBE OAUTH AUTHENTICATION")
        print("="*60)
        print("1. Open the following URL in your web browser (phone or laptop):")
        
        # Try local server flow with fallback
        try:
            creds = flow.run_local_server(
                bind_addr="0.0.0.0",
                port=port,
                open_browser=False,
                prompt="consent"
            )
        except Exception as err:
            print(f"\nLocal server flow failed: {err}")
            print("Falling back to console auth...")
            auth_url, _ = flow.authorization_url(prompt="consent")
            print(f"\nAuth URL:\n{auth_url}\n")
            code = input("Enter the authorization code (or full redirected URL): ").strip()
            if "code=" in code:
                import urllib.parse
                parsed = urllib.parse.urlparse(code)
                code = urllib.parse.parse_qs(parsed.query).get("code", [code])[0]
            flow.fetch_token(code=code)
            creds = flow.credentials

        # Save credentials for next time
        with open(token_file, "w") as f:
            f.write(creds.to_json())
        print(f"✅ Credentials saved to {token_file}")

    return build("youtube", "v3", credentials=creds)


def upload_video(youtube, video_path, title, description, tags, privacy="public"):
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError

    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        sys.exit(1)

    print(f"\n📤 Uploading '{video_path}' ({os.path.getsize(video_path)/1e6:.2f} MB) to YouTube Shorts...")
    print(f"Title: {title}")
    print(f"Privacy: {privacy}")

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": "28"  # Science & Technology
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False
        }
    }

    media = MediaFileUpload(
        video_path,
        mimetype="video/mp4",
        chunksize=1024*1024*2,
        resumable=True
    )

    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media
    )

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"  Uploading... {int(status.progress() * 100)}%")

    video_id = response.get("id")
    url = f"https://youtube.com/shorts/{video_id}"
    print(f"\n🎉 UPLOAD SUCCESSFUL!")
    print(f"Video ID: {video_id}")
    print(f"Shorts URL: {url}")
    return url


def main():
    parser = argparse.ArgumentParser(description="Upload Microduck YouTube Short")
    parser.add_argument("--secrets", default="client_secrets.json", help="Path to Google client_secrets.json")
    parser.add_argument("--token", default="youtube_token.json", help="Path to save/load auth token")
    parser.add_argument("--privacy", default="public", choices=["public", "unlisted", "private"], help="Video privacy")
    parser.add_argument("--video", default="/home/ubuntu/vibeduck/youtube_shorts_microduck_tail_wag.mp4", help="Video file path")
    parser.add_argument("--port", type=int, default=8080, help="Local OAuth server port")
    args = parser.parse_args()

    title = "Teaching a Robot Duck to Tail Wag with AI (9 Attempts!) 🦆🤖 #Shorts"
    
    description = (
        "Teaching our 15-servo Microduck biped robot how to do a duck butt-wiggle (\"preen shake\") "
        "using Reinforcement Learning (PPO) in MuJoCo.\n\n"
        "Attempt 1: Backward somersault face-plant crash 💥\n"
        "Attempts 2-5: Duck discovers standing frozen avoids penalties (peak AI laziness) 🧊\n"
        "Attempts 6-7: Tries to move, trips on its own webbed feet 👣\n"
        "Attempt 8: The 3.2 Hz breakthrough (clean wiggle, but conservative amplitude) ★\n"
        "Attempt 9: Multiplicative Pythagorean reward = VIGOROUS 3.0 Hz duck butt-wiggle with locked forward gaze! 🦆✨\n\n"
        "🎵 Music: Kevin MacLeod - 'Fluffing a Duck' (122 BPM)\n"
        "Trained on NVIDIA L4 GPU via Modal.\n"
        "Code & ONNX models: https://github.com/WoSupport/microduck_rl\n\n"
        "#Microduck #Robotics #ReinforcementLearning #AI #BipedRobot #MuJoCo #Shorts #Tech"
    )
    
    tags = [
        "microduck", "robotics", "reinforcement learning", "biped robot", 
        "AI", "sim2real", "mujoco", "tail wag", "shorts", "duck robot", "robot dance"
    ]

    youtube = get_authenticated_service(secrets_file=args.secrets, token_file=args.token, port=args.port)
    upload_video(youtube, args.video, title, description, tags, privacy=args.privacy)


if __name__ == "__main__":
    main()
