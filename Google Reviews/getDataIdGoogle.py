from dotenv import load_dotenv
import os
import requests

load_dotenv()

#TODO check of dit met geocoding beter werkt of niet
def get_place_id(name, address):
    base_url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
    params = {
        "input": f"{name}, {address}",
        "inputtype": "textquery",
        "fields": "place_id",
        "key": os.getenv("GOOGLE_API_KEY")
    }

    response = requests.get(base_url, params=params)
    result = response.json()

    if result.get("candidates"):
        return result["candidates"][0]["place_id"]
    else:
        return None

#print(get_place_id("Vrije Universiteit Brussel", "Pleinlaan 2"))
#print(os.getenv("GOOGLE_API_KEY"))