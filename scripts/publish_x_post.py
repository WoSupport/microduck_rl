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

TWEET_1 = """We spent 9 attempts using Reinforcement Learning to teach our 15-servo biped duck robot how to wag its tail.

The result is pure AI comedy and ultimate triumph:
• Attempt 1: Backward somersault faceplant 💥
• Attempts 2-5: Camping in "freeze" mode to farm points 🧊
• Attempt 8: 3.2 Hz wiggle unlocked! ★
• Attempt 9: THE FINAL GLORIOUS TAIL WAG 🦆✨

Full journey breakdown below 👇 #Robotics #AI"""

TWEET_2 = """Why was this trick so tricky to learn?

When we used standard additive reward tracking, the RL optimizer discovered a hilarious "compromise basin":
If the duck stands completely frozen, it avoids falling penalties and collects steady points with 0% risk. Peak robot laziness.

To break the freeze, we switched to a multiplicative Pythagorean formulation where standing still yields identically 0.0000 reward. Move or starve! ⚡"""

TWEET_3 = """The Final Policy (Attempt 9):
🦆 3.0 Hz vigorous lateral tail-wag / preen shake
👀 Gaze-locked forward stabilization (vestibulo-ocular reflex)
👣 Webbed feet glued to the floor with zero drift or falls
⚡ 776 KB standalone ONNX policy running at 50 Hz on Dynamixel XL330 servos

Trained on NVIDIA L4 GPU via Modal.
Code & models are open source:
https://github.com/WoSupport/microduck_rl"""


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
            media_category="tweet_video"
        )
        media_id = media.media_id_string
        print(f"✅ Video uploaded! Media ID: {media_id}")
        
        # Check processing status for video
        print("Waiting for Twitter video processing...")
        while True:
            status = api_v1.get_media_upload_status(media_id)
            state = status.processing_info.get("state")
            if state == "succeeded":
                print("✅ Video processing completed!")
                break
            elif state == "failed":
                print(f"❌ Video processing failed: {status.processing_info}")
                sys.exit(1)
            else:
                wait_secs = status.processing_info.get("check_after_secs", 5)
                print(f"  Processing... waiting {wait_secs}s")
                time.sleep(wait_secs)

    # Post Tweet 1 (with video)
    print("\n🚀 Posting Main Tweet...")
    kwargs = {"text": TWEET_1}
    if media_id:
        kwargs["media_ids"] = [media_id]
        
    res1 = client_v2.create_tweet(**kwargs)
    t1_id = res1.data["id"]
    print(f"✅ Main Tweet published! https://x.com/{user.screen_name}/status/{t1_id}")

    # Post Tweet 2 (thread reply)
    time.sleep(2)
    print("\n🧵 Posting Thread Reply 1...")
    res2 = client_v2.create_tweet(text=TWEET_2, in_reply_to_tweet_id=t1_id)
    t2_id = res2.data["id"]
    print(f"✅ Reply 1 published! https://x.com/{user.screen_name}/status/{t2_id}")

    # Post Tweet 3 (thread reply)
    time.sleep(2)
    print("\n🧵 Posting Thread Reply 2...")
    res3 = client_v2.create_tweet(text=TWEET_3, in_reply_to_tweet_id=t2_id)
    t3_id = res3.data["id"]
    print(f"✅ Reply 2 published! https://x.com/{user.screen_name}/status/{t3_id}")

    print("\n🎉 ENTIRE THREAD PUBLISHED SUCCESSFULLY ON X!")
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
