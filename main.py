import pandas as pd
import sys

from getAddressAndNameByKbo import fetch_kbo_data
sys.path.append("Google Reviews")
from getDataIdGoogle import get_place_id
from getDetailsFromPlacesApi import get_details_from_places_api

# with kbonummers adding address and company name (Dit voorbeeld met VUB, philip morris & café belga)
data = pd.read_csv('testKboNummers.csv', delimiter=',')
data[['Name', 'Address']] = data['ondernemingsnummer'].apply(lambda num: pd.Series(fetch_kbo_data(num)))

# adding place_id from google
data['place_id'] = data.apply(lambda row: get_place_id(row['Name'], row['Address']), axis=1)

# Getting Google reviews
data['reviews'] = data.apply(lambda row: get_details_from_places_api(row['place_id']), axis=1)

data
## best way to show dataframe
