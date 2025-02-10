from google_maps_reviews import ReviewsClient

api_client = ReviewsClient(api_key='NDFkMWExYTIzMDY5NDkwNTkzMDMzOTlhODNjYTg2YWV8ZDZkZmY2MjA0ZQ')
results = api_client.get_reviews('ChIJQTNrM69ZwokR3ggxzgeelqQ', reviewsLimit=250, language='en')


print(results)