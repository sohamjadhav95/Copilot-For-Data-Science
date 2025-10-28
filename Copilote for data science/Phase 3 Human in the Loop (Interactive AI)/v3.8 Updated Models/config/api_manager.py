# config/api_manager.py
import os
import json
from datetime import datetime, timedelta

CONFIG_DIR = "config"
KEY_FILE = os.path.join(CONFIG_DIR, "api_key.txt")
EXPIRY_DAYS = 7

def get_api_key():
    # Ensure config folder exists
    os.makedirs(CONFIG_DIR, exist_ok=True)

    # Check if file exists and load content
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "r") as f:
            try:
                data = json.load(f)
                api_key = data.get("key", "")
                saved_time = datetime.fromisoformat(data.get("saved_on", "1970-01-01"))
                if datetime.now() - saved_time < timedelta(days=EXPIRY_DAYS):
                    return api_key
                else:
                    print("🔁 API key expired after 7 days.")
            except Exception:
                print("⚠️ Failed to parse existing API key file.")
    
    # Ask for new key
    new_key = input("🔐 Enter your Groq API key: ").strip()
    with open(KEY_FILE, "w") as f:
        json.dump({
            "key": new_key,
            "saved_on": datetime.now().isoformat()
        }, f)
    print("✅ API key saved and will expire in 7 days.")
    return new_key
