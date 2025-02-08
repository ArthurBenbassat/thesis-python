import requests

def getId():
    url = "https://www.searchapi.io/api/v1/search"
    params = {
        "engine": "google_maps",
        "q": "Scouts 85ste Elsdonk",
        "ll": "@51.1666178,4.4260553,16z",
        "api_key": ""
    }

    response = requests.get(url, params=params)
    print(response.text)


def getGoogleReviewsByAddressAndName(place_id):
    url = "https://www.searchapi.io/api/v1/search"
    params = {
        "engine": "google_maps_reviews",
        "place_id": place_id,
        "api_key": "2rD7dNX8qmYSyzABi9bQEE5i"
    }

    response = requests.get(url, params=params)
    print(response.text)

#getId()
getGoogleReviewsByAddressAndName("ChIJKxpH4MjEw0cRxX8vf-QApzQ")