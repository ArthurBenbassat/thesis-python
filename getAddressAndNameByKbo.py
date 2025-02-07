import requests
import pandas as pd
url = "https://kbopub.economie.fgov.be/kbopub/zoeknummerform.html"

data = pd.read_csv('testKboNummers.csv', delimiter=',')
print(data.head())

for index, row in data.iterrows():
    request = requests.get(url, params={'nummer': row['ondernemingsnummer']})
    print(request.status_code)
#requests.get(url, params={nummer: value, }, args)
