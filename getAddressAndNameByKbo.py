import requests
import pandas as pd
from bs4 import BeautifulSoup

url = "https://kbopub.economie.fgov.be/kbopub/zoeknummerform.html"

data = pd.read_csv('testKboNummers.csv', delimiter=',')



def fetch_kbo_data(ondernemingsnummer):
    response = requests.get(url, params={'nummer': ondernemingsnummer})

    if response.status_code != 200:
        return pd.NA, pd.NA

    soup = BeautifulSoup(response.text, "lxml")

    # Check of het nummer ongeldig is
    if "Het nummer is ongeldig." in soup.text:
        return pd.NA, pd.NA

    # Get company name from
    naam_tag = soup.find("td", string="Naam:")
    naam = naam_tag.find_next("td").find(text=True, recursive=False).strip() if naam_tag else pd.NA

    # Get Address
    adres_tag = soup.find("td", string="Adres van de zetel:")
    adres = adres_tag.find_next("td").text.strip().replace("\n", " ").replace("\xa0", " ") if adres_tag else pd.NA

    return naam, adres


data[['Naam', 'Adres']] = data['ondernemingsnummer'].apply(lambda num: pd.Series(fetch_kbo_data(num)))


print(data.head())