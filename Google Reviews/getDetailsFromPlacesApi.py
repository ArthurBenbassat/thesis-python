from dotenv import load_dotenv
import os
import requests

load_dotenv()
def get_details_from_places_api(place_id):
    base_url = 'https://maps.googleapis.com/maps/api/place/details/json'
    params = {
        "place_id": place_id ,
        "fields": "reviews",
        "key": os.getenv("GOOGLE_API_KEY")
    }

    response = requests.get(base_url, params=params)
    return response.json()

print()