#!/usr/bin/env python3
"""
Publish Microduck Tail Wag Progression to X (Twitter).
Supports:
1. Automated posting via X Developer API v2 + v1.1 chunked video upload (Tweepy).
2. Generating a ready-to-use 1-click Web Intent link + formatted thread text.
"""

import os
import sys
import json
import time
import urllib.parse
import argparse

SINGLE_POST = """9 attempts using RL to teach the @pollenrobotics Microduck biped how to wag its tail:

• Att 1: Somersault crash 💥
• Att 2-5: 'Freeze' mode camping 🧊
• Att 8: 3.2 Hz wiggle! ★
• Att 9: VIGOROUS TAIL WAG 🦆✨

50 Hz ONNX ready for hardware! #Robotics #AI"""

TWEET_1 = SINGLE_POST
TWEET_2 = ""
TWEET_3 = ""


def get_credentials(creds_file="x_credentials.json"):
    # First check env vars
    api_key = os.environ.get("X_API_KEY")
    api_secret = os.environ.get("X_API_SECRET")
    access_token = os.environ.get("X_ACCESS_TOKEN")
    access_token_secret = os.environ.get("X_ACCESS_TOKEN_SECRET")

    if api_key and api_secret and access_token and access_token_secret:
        return {
            "api_key": api_key,
            "api_secret": api_secret,
            "access_token": access_token,
            "access_token_secret": access_token_secret
        }

    # Then check json file
    if os.path.exists(creds_file):
        with open(creds_file, "r") as f:
            data = json.load(f)
            return {
                "api_key": data.get("api_key") or data.get("consumer_key"),
                "api_secret": data.get("api_secret") or data.get("consumer_secret"),
                "access_token": data.get("access_token"),
                "access_token_secret": data.get("access_token_secret")
            }

    return None


def post_via_api(creds, video_path):
    import tweepy

    print("\n🔐 Authenticating with X API...")
    auth = tweepy.OAuth1UserHandler(
        creds["api_key"],
        creds["api_secret"],
        creds["access_token"],
        creds["access_token_secret"]
    )
    api_v1 = tweepy.API(auth)
    client_v2 = tweepy.Client(
        consumer_key=creds["api_key"],
        consumer_secret=creds["api_secret"],
        access_token=creds["access_token"],
        access_token_secret=creds["access_token_secret"]
    )

    # Verify credentials
    user = api_v1.verify_credentials()
    print(f"✅ Authenticated as @{user.screen_name} ({user.name})")

    media_id = None
    if video_path and os.path.exists(video_path):
        print(f"\n📤 Uploading video '{video_path}' ({os.path.getsize(video_path)/1e6:.2f} MB) via chunked upload...")
        media = api_v1.media_upload(
            filename=video_path,
            chunked=True,
            media_category="tweet_video",
            wait_for_async_finalize=True
        )
        media_id = media.media_id_string
        print(f"✅ Video uploaded & processed! Media ID: {media_id}")

    # Post Tweet 1 (with video)
    print("\n🚀 Posting Main Tweet...")
    kwargs = {"text": TWEET_1}
    if media_id:
        kwargs["media_ids"] = [media_id]
        
    res1 = client_v2.create_tweet(**kwargs)
    t1_id = res1.data["id"]
    print(f"✅ Main Tweet published! https://x.com/{user.screen_name}/status/{t1_id}")

    print(f"\n🎉 POST PUBLISHED SUCCESSFULLY ON X!")
    print(f"👉 URL: https://x.com/{user.screen_name}/status/{t1_id}")
    return f"https://x.com/{user.screen_name}/status/{t1_id}"


def print_manual_instructions():
    intent_url = "https://twitter.com/intent/tweet?text=" + urllib.parse.quote(TWEET_1)
    print("\n" + "="*60)
    print("📝 1-CLICK WEB INTENT & COPY-PASTE THREAD")
    print("="*60)
    print(f"\n1-Click Pre-filled Tweet Intent Link:\n{intent_url}\n")
    print("Video File to Attach:\n/home/ubuntu/vibeduck/youtube_shorts_microduck_tail_wag.mp4")
    print("Direct Web Download Link for Phone/Laptop:\nhttps://manchester-partner-inline-sep.trycloudflare.com/youtube_shorts_microduck_tail_wag.mp4\n")
    print("-" * 60)
    print("TWEET 1 (Main Post with Video):")
    print(TWEET_1)
    print("-" * 60)
    print("TWEET 2 (Reply 1):")
    print(TWEET_2)
    print("-" * 60)
    print("TWEET 3 (Reply 2):")
    print(TWEET_3)
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Publish Microduck Thread to X")
    parser.add_argument("--creds", default="x_credentials.json", help="Path to x_credentials.json")
    parser.add_argument("--video", default="/home/ubuntu/vibeduck/youtube_shorts_microduck_tail_wag.mp4", help="Video file path")
    parser.add_argument("--manual", action="store_true", help="Print thread text & 1-click intent link")
    args = parser.parse_args()

    if args.manual:
        print_manual_instructions()
        return

    creds = get_credentials(args.creds)
    if not creds:
        print("❌ No X API credentials found in environment or x_credentials.json.")
        print("Run with --manual to see 1-click posting instructions, or set up x_credentials.json.")
        print_manual_instructions()
        sys.exit(1)

    post_via_api(creds, args.video)


if __name__ == "__main__":
    main()
