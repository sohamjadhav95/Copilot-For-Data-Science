# config/requirement_checker.py
import os
import subprocess
import json
from datetime import datetime, timedelta

CHECK_FILE = os.path.join("config", "last_checked.json")
REQUIREMENTS_FILE = "requirements.txt"
EXPIRY_DAYS = 7


def is_check_expired():
    if not os.path.exists(CHECK_FILE):
        return True
    
    try:
        with open(CHECK_FILE, "r") as f:
            data = json.load(f)
            last_checked = datetime.fromisoformat(data.get("last_checked", "1970-01-01"))
            return datetime.now() - last_checked > timedelta(days=EXPIRY_DAYS)
    except Exception:
        return True


def update_check_timestamp():
    with open(CHECK_FILE, "w") as f:
        json.dump({"last_checked": datetime.now().isoformat()}, f)


def install_requirements():
    print("📦 Installing required packages from requirements.txt...")
    subprocess.check_call(["python", "-m", "pip", "install", "-r", REQUIREMENTS_FILE])
    print("✅ All requirements installed successfully.")


def check_and_install_requirements():
    if is_check_expired():
        print("🔁 Rechecking dependencies (7-day refresh)...")
        install_requirements()
        update_check_timestamp()
    else:
        print("🟢 Requirements already checked recently. Skipping reinstall.")


# Optional: Run on import
if __name__ == "__main__":
    check_and_install_requirements()
