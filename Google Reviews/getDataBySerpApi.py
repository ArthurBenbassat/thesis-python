from dotenv import load_dotenv
import os
from serpapi.google_search import GoogleSearch

load_dotenv()
def get_data_by_serp_api():
  params = {
    "engine": "google_maps_reviews",
    "data_id": "0x89c259af336b3341:0xa4969e07ce3108de",
    #"hl": "fr",
    "api_key": os.getenv("SERP_API_KEY"),
  }

  search = GoogleSearch(params)
  results = search.get_dict()
  print(results)

def get_next_page(place_id, page_token):
  params = {
    "engine": "google_maps_reviews",
    "data_id": place_id,
    "page_token": page_token,

  }



get_data_by_serp_api()