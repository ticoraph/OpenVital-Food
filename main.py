import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#df = pd.read_parquet("hf://datasets/openfoodfacts/open-prices/prices.parquet")

df = pd.read_parquet("dataset/prices.parquet")

#print(df.head())
#print(df.describe())
#print(df.info())

print(df.groupby("type").describe())
# price
# currency : EUR
# updated : date
# location_osm_address_country
# location_osm_address_country_code
# location_osm_display_name
# product_code
# product_name
# price_is_discounted
# price_without_discount

eu_countries = [
    'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR',
    'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL',
    'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE'
]

country_translation = {
    'België / Belgique / Belgien': 'Belgique',
    'Deutschland': 'Allemagne',
    'Eesti': 'Estonie',
    'España': 'Espagne',
    'France': 'France',
    'Hrvatska': 'Croatie',
    'Italia': 'Italie',
    'Lietuva': 'Lituanie',
    'Nederland': 'Pays-Bas',
    'Portugal': 'Portugal',
    'Suomi / Finland': 'Finlande',
    'Österreich': 'Autriche',
    'Ελλάς': 'Grèce'
}

df['location_osm_address_country'] = (
    df['location_osm_address_country']
    .replace(country_translation)
)

# Supprimer les lignes

df = df[(df['price'] > 0)]
df['price'] = df['price'].astype(float)
df['price_without_discount'] = df['price_without_discount'].astype(float)

##################################################
    
# Filtrer pour EUR et pays UE, puis grouper par pays
df_eur = df[
    (df['currency'] == 'EUR') &
    (df['location_osm_address_country_code'].isin(eu_countries))
].groupby('location_osm_address_country')['price'].agg([
    'mean',   # Prix moyen
    'median', # Prix médian
    'min',    # Prix minimum
    'max',    # Prix maximum
    'count'   # Nombre de produits
]).reset_index()
df_eur = df_eur[df_eur['count'] >= 20]
print(df_eur)

##################################################

def analyze_iqr(data):
    """
    Analyse la distribution des données et identifie les valeurs aberrantes (outliers)
    en utilisant la méthode de l'écart interquartile (IQR).
    Args:
        data: Liste ou array de valeurs numériques à analyser
    Returns:
        outliers: Liste des valeurs considérées comme aberrantes
    """

    # Calcul des quartiles
    q1 = np.percentile(data, 25)  # Premier quartile (25% des données)
    #q2 = np.percentile(data, 50)  # Médiane (50% des données)
    q3 = np.percentile(data, 75)  # Troisième quartile (75% des données)
    iqr = q3 - q1  # Écart interquartile (mesure de dispersion)

    # Définition des bornes pour détecter les outliers
    # Utilisation d'un facteur de 2 au lieu de 1.5 (méthode moins stricte)
    lower_bound = q1 - 1.5 * iqr  # Borne inférieure
    upper_bound = q3 + 1.5 * iqr  # Borne supérieure

    # Identification des valeurs aberrantes (en dehors des bornes)
    outliers = [x for x in data if x < lower_bound or x > upper_bound]

    return outliers

#############################################
# VISUALISATION INITIALE : Boxplots avant nettoyage des données
#############################################

df_eur = df[
    (df['currency'] == 'EUR') &
    (df['location_osm_address_country_code'].isin(eu_countries))
].copy()


plt.figure(figsize=(18, 8))

# Affichage de 8 variables sous forme de boxplots pour identifier visuellement les outliers
plt.subplot(2, 4, 1)
plt.boxplot(df_eur['price'])
plt.title('price')
plt.tight_layout()
plt.show()

#############################################
# NETTOYAGE DES DONNÉES : Correction des valeurs aberrantes et suppression des outliers
#############################################

# Électricité : nettoyage en 3 étapes
price = analyze_iqr(df_eur['price'])  # 1. Identification des outliers via IQR
df_eur.drop(df_eur[df_eur["price"].isin(price)].index, inplace=True)  # 2. Suppression des outliers identifiés

#############################################
# VISUALISATION FINALE : Boxplots après nettoyage des données
#############################################

plt.figure(figsize=(18, 8))
# Affichage des mêmes variables après traitement pour vérifier l'efficacité du nettoyage
plt.subplot(2, 4, 1)
plt.boxplot(df_eur['price'])
plt.title('price')
plt.tight_layout()
plt.show()

##################################################

# Calcul du prix moyen par pays
mean_price = (
    df_eur.groupby('location_osm_address_country')['price']
    .mean()
    .sort_values(ascending=True)
)

# Tracé
plt.figure(figsize=(14, 6))
plt.bar(mean_price.index, mean_price.values)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Prix moyen (€)')
plt.xlabel('Pays')
plt.title('Prix moyen par pays (EUR)')
plt.tight_layout()
plt.show()