import requests
import pandas as pd
from bs4 import BeautifulSoup
import re

url = "https://kbopub.economie.fgov.be/kbopub/zoeknummerform.html"

data = pd.read_csv('testKboNummers.csv', delimiter=',')

def extract_before_sinds(text):
    match = re.match(r"^(.*?)\s*Sinds", text)
    return match.group(1)


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
    adres = extract_before_sinds(adres_tag.find_next("td").text.strip().replace("\n", " ").replace("\xa0", " ").replace("\t", "")) if adres_tag else pd.NA

    return naam, adres


data[['Naam', 'Adres']] = data['ondernemingsnummer'].apply(lambda num: pd.Series(fetch_kbo_data(num)))


# print first row of data
print(data[['Naam', 'Adres']].head())