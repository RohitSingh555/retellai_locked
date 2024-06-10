import requests
import os
from dotenv import load_dotenv
load_dotenv(override=True)

retell_api_key = os.environ['RETELL_API_KEY']

def get_call_details(call_id, token):
    url = f"https://api.retellai.com/get-call/{call_id}"
    headers = {
        'Authorization': f'Bearer {token}'
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()


# This is how you can use it to get transcript of any call id which you can find on retell dashboard history tab
call_id = "921ad7d1d911eae54dda2835b46345f5"
# token taken from retellai
token = retell_api_key

try:
    call_details = get_call_details(call_id, token)
    # to get only the transcript information
    print(call_details.get('transcript'))
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")