import pandas as pd

# Namen van de input CSV-bestanden en het output CSV-bestand
eerste_bestand = 'Data/Outscraper-non-fraud.csv'
tweede_bestand = 'Data/__Outscraper_20250506084914m97_.csv'
output_bestand = 'Data/combined_with_reviews.csv'

# Lijst met de kolommen die we willen behouden
gewenste_kolommen = [
    'place_id',
    'reviews',
    'rating',
    'review_id',
    'author_id',
    'review_text',
    'owner_answer',
    'owner_answer_timestamp_datetime_utc',
    'review_link',
    'review_rating',
    'review_timestamp',
    'review_datetime_utc',
    'review_likes',
    'author_reviews_count',
    'author_ratings_count',
    'review_questions_Positive',
    'review_questions_Negative'
]

# --- Verwerking van het eerste bestand ---
try:
    df_eerste = pd.read_csv(eerste_bestand, low_memory=False)  # low_memory=False toegevoegd
    print("Kolomnamen in het eerste bestand:", df_eerste.columns.tolist()) # Print de kolomnamen
    # Voeg de 'fraud' kolom toe met de waarde 0 als de eerste kolom
    df_eerste.insert(0, 'fraud', 0)
    # Selecteer de gewenste kolommen
    df_eerste = df_eerste[gewenste_kolommen + ['fraud']]
    # Verwijder rijen waar review_id '__NO_REVIEWS_FOUND__' is
    df_eerste = df_eerste[df_eerste['review_id'] != '__NO_REVIEWS_FOUND__']
except FileNotFoundError:
    print(f"Fout: Het bestand '{eerste_bestand}' is niet gevonden.")
    exit()

# --- Verwerking van het tweede bestand ---
try:
    df_tweede = pd.read_csv(tweede_bestand)
    # Voeg de 'fraud' kolom toe met de waarde 1 als de eerste kolom
    df_tweede.insert(0, 'fraud', 1)
    # Selecteer de gewenste kolommen
    df_tweede = df_tweede[gewenste_kolommen + ['fraud']]
    # Verwijder rijen waar review_id '__NO_REVIEWS_FOUND__' is
    df_tweede = df_tweede[df_tweede['review_id'] != '__NO_REVIEWS_FOUND__']
except FileNotFoundError:
    print(f"Fout: Het bestand '{tweede_bestand}' is niet gevonden.")
    exit()

# --- Samenvoegen van de twee DataFrames ---
df_gecombineerd = pd.concat([df_eerste, df_tweede], ignore_index=True)

# --- Opslaan naar een nieuw CSV-bestand ---
try:
    df_gecombineerd.to_csv(output_bestand, index=False)
    print(f"De bestanden zijn succesvol gecombineerd en opgeslagen als '{output_bestand}'.")
except Exception as e:
    print(f"Er is een fout opgetreden bij het opslaan van het bestand: {e}")