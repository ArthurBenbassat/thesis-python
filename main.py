import pandas as pd
import sys

from getAddressAndNameByKbo import fetch_kbo_data
sys.path.append("Google Reviews")
from getDataIdGoogle import get_place_id
from getDetailsFromPlacesApi import get_details_from_places_api


data = pd.read_csv('Data/misdrijf_pubs.csv', delimiter=',')

# get only the first 10 rows for data
data = data.head(10)

data[['Name', 'Address']] = data['md_VAT'].apply(lambda num: pd.Series(fetch_kbo_data(num)))

# adding place_id from google
data['place_id'] = data.apply(lambda row: get_place_id(row['Name'], row['Address']), axis=1)

# Getting Google reviews
#data['reviews'] = data.apply(lambda row: get_details_from_places_api(row['place_id']), axis=1)

data

