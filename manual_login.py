"""
manual_login.py
===============
Run this ONCE to log into LinkedIn manually.
Your session will be saved to the persistent Chrome profile
so the scraper can reuse it without logging in again.

Usage:
    python3 manual_login.py
"""

import os
import time
import undetected_chromedriver as uc

# Same profile directory used by the scraper
PROFILE_DIR = os.path.join(os.path.expanduser("~"), "chrome_linkedin_profile")
os.makedirs(PROFILE_DIR, exist_ok=True)

print(f"📁  Using Chrome profile: {PROFILE_DIR}")
print("🌐  Opening LinkedIn login page...")
print("👉  Log in manually in the browser window.")
print("⏳  Script will wait 60 seconds for you to complete login.\n")

options = uc.ChromeOptions()
options.add_argument(f"--user-data-dir={PROFILE_DIR}")
options.add_argument("--window-size=1280,900")

# Open in headed mode so you can type your credentials. Let UC auto-detect
# the installed Chrome version instead of pinning an old major version.
driver = uc.Chrome(options=options, use_subprocess=True)

# Go to LinkedIn login page
driver.get("https://www.linkedin.com/login")

# Wait for you to log in manually
print("⏳  Waiting 60 seconds — log in now...")
time.sleep(60)

# Check if login succeeded
if "feed" in driver.current_url or "mynetwork" in driver.current_url:
    print("✅  Login successful! Session saved to Chrome profile.")
    print("    You can now run: streamlit run app.py")
    print("    The scraper will use this session automatically.")
else:
    print("⚠️  Login may not have completed.")
    print(f"    Current URL: {driver.current_url}")
    print("    Try running this script again and log in faster.")

driver.quit()
print("\n🎉  Done! Chrome profile saved.")
