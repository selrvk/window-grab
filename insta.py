from instagrapi import Client
import time

# 1. Initialize and Login
cl = Client()
cl.login("selrvk", "vanessa1105")

TARGET_USERNAME = "pythontes7ing"
target_user_id = cl.user_id_from_username(TARGET_USERNAME)

print(f"Monitoring {TARGET_USERNAME}...")

while True:
    try:
        # 2. Check for active broadcasts
        broadcasts = cl.user_live_broadcast(target_user_id)
        
        if broadcasts:
            print(f"ALERT: {TARGET_USERNAME} is LIVE!")
            # Add code here to send yourself a Telegram message or email
        else:
            print("Not live yet...")
            
    except Exception as e:
        print(f"Error occurred: {e}")

    # 3. Wait before checking again (don't get banned!)
    time.sleep(300) # Check every 5 minutes