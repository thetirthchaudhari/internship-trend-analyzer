"""
manual_naukri_login.py
======================
Run this ONCE to log into Naukri.com and save the session
to ~/chrome_naukri_profile/ — the scraper reuses this profile
on every subsequent run, so no repeated logins are needed.

Usage:
    source venv/bin/activate
    python manual_naukri_login.py
"""

import os
import time
import undetected_chromedriver as uc

PROFILE_DIR = os.path.join(os.path.expanduser("~"), "chrome_naukri_profile")
os.makedirs(PROFILE_DIR, exist_ok=True)

opts = uc.ChromeOptions()
opts.add_argument(f"--user-data-dir={PROFILE_DIR}")
opts.add_argument("--window-size=1280,900")

print(f"\n[INFO] Opening Chrome with profile: {PROFILE_DIR}")
print("[INFO] Please log into Naukri.com manually in the browser window.")
print("[INFO] After logging in successfully, press ENTER here to save and close.\n")

# Let undetected_chromedriver match the locally installed Chrome version.
driver = uc.Chrome(options=opts, use_subprocess=True)
driver.get("https://www.naukri.com/nlogin/login")
time.sleep(3)

input(">>> Log in to Naukri.com in the browser, then press ENTER to save session: ")

print("[INFO] Session saved to:", PROFILE_DIR)
print("[INFO] You can now run the Naukri scraper — no login required.\n")
driver.quit()
