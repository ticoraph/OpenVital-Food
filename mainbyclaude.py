import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import List, Dict
import seaborn as sns
import unidecode

# Configuration matplotlib pour de meilleurs graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class PriceAnalyzer:
    """Analyseur de prix pour les données Open Food Facts"""

    EU_COUNTRIES = [
        'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR',
        'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL',
        'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE'
    ]

    COUNTRY_TRANSLATION = {
        'België / Belgique / Belgien': 'Belgique',
        'Belgique': 'Belgique',
        'Deutschland': 'Allemagne',
        'Allemagne': 'Allemagne',
        'Eesti': 'Estonie',
        'Estonie': 'Estonie',
        'España': 'Espagne',
        'Espagne': 'Espagne',
        'France': 'France',
        'Hrvatska': 'Croatie',
        'Croatie': 'Croatie',
        'Italia': 'Italie',
        'Italie': 'Italie',
        'Lietuva': 'Lituanie',
        'Lituanie': 'Lituanie',
        'Nederland': 'Pays-Bas',
        'Pays-Bas': 'Pays-Bas',
        'Portugal': 'Portugal',
        'Suomi / Finland': 'Finlande',
        'Finlande': 'Finlande',
        'Österreich': 'Autriche',
        'Autriche': 'Autriche',
        'Ελλάς': 'Grèce',
        'Grèce': 'Grèce',
        'Slovenija': 'Slovénie',
        'Slovénie': 'Slovénie',
        'Éire / Ireland': 'Irlande',
        'Ireland': 'Irlande',
        'Irlande': 'Irlande',
        'Slovensko': 'Slovaquie',
        'Slovaquie': 'Slovaquie',
        'Lëtzebuerg': 'Luxembourg',
        'Luxembourg': 'Luxembourg',
        'Κύπρος - Kıbrıs': 'Chypre',
        'Chypre': 'Chypre',
        'Latvija': 'Lettonie',
        'Lettonie': 'Lettonie',
        'България': 'Bulgarie',
        'Bulgarie': 'Bulgarie'
    }

    CITY_TO_REMOVE = {
        "saint-martin-d'heres",
        "echirolles",
        "lyon",
        "universite",
        "villeurbanne",
        "part-dieu",
        "grenoble",
        "meylan",
        "levallois",
        "bresson",
        "port"
    }

    def __init__(self, filepath: str):
        """
        Initialise l'analyseur avec le chemin du fichier de données

        Args:
            filepath: Chemin vers le fichier parquet
        """
        self.df = pd.read_parquet(filepath)
        self.df_cleaned = None

    def clean_data(self) -> pd.DataFrame:
        """
        Nettoie et prépare les données pour l'analyse

        Returns:
            DataFrame nettoyé
        """
        df = self.df.copy()

        # Traduction des noms de pays
        df['location_osm_address_country'] = (
            df['location_osm_address_country'].replace(self.COUNTRY_TRANSLATION)
        )

        # Filtrage et conversion des types
        df = df[df['price'] > 0].copy()
        df['price'] = df['price'].astype(float)
        df['price_without_discount'] = pd.to_numeric(
            df['price_without_discount'], errors='coerce'
        )

        # Conversion de la colonne updated en datetime
        df['updated'] = pd.to_datetime(df['updated'], errors='coerce')

        self.df_cleaned = df
        return df

    def clean_store(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie et prépare les données pour l'analyse des magasins/enseignes.
        Normalise les noms et retire les lignes correspondant aux villes génériques listées.
        """
        df = df.copy()

        # Extraction du nom du magasin (première partie avant la virgule)
        df['osm_name'] = df['location_osm_display_name'].str.split(',').str[0].astype(str)

        # 1) Supprimer les accents
        df['osm_name'] = df['osm_name'].apply(lambda x: unidecode.unidecode(x) if isinstance(x, str) else x)

        # 2) Mettre en minuscules
        df['osm_name'] = df['osm_name'].str.lower()

        # 3) Normaliser la ponctuation : garder lettres, chiffres, espaces et tirets
        #    remplacer tout le reste par un espace, puis compacter les espaces et strips
        df['osm_name_norm'] = df['osm_name'].apply(
            lambda x: re.sub(r'[^a-z0-9\-\s]', ' ', x) if isinstance(x, str) else x
        )
        df['osm_name_norm'] = df['osm_name_norm'].str.replace(r'\s+', ' ', regex=True).str.strip()
        # (optionnel) remplacer espaces par tirets si tu préfères format avec tirets
        df['osm_name_norm'] = df['osm_name_norm'].str.replace(' ', ' ', regex=False)  # laisse tel quel

        # 4) Préparer la liste normalisée CITY_TO_REMOVE (une seule fois pour la classe serait mieux)
        #    On normalise les éléments de CITY_TO_REMOVE de la même façon pour comparaison fiable
        def normalize_token(tok: str) -> str:
            t = unidecode.unidecode(tok).lower()
            t = re.sub(r'[^a-z0-9\-\s]', ' ', t)
            t = re.sub(r'\s+', ' ', t).strip()
            return t

        city_norm_set = {normalize_token(c) for c in self.CITY_TO_REMOVE}

        # 5) Filtrer : on supprime les lignes où osm_name_norm est exactement dans city_norm_set
        mask_exact = df['osm_name_norm'].isin(city_norm_set)

        # 6) Aussi supprimer si le nom est composé uniquement d'un des tokens (ex: "grenoble hypermarché")
        #    Ici on va considérer la première "mot-clé" (avant espace) ou check de mot isolé avec regex word-boundary
        pattern = r'\\b(?:' + '|'.join(re.escape(c) for c in city_norm_set if c) + r')\\b'
        # build a regex with real word boundaries (note: python string needs single backslashes)
        pattern = r'\b(?:' + '|'.join(re.escape(c) for c in city_norm_set if c) + r')\b'
        mask_word = df['osm_name_norm'].str.contains(pattern, regex=True, na=False)

        # Combine masks (on retire si exact OU contient le token isolé)
        mask_remove = mask_exact | mask_word

        # Debug: afficher combien on supprime
        removed_count = mask_remove.sum()
        total = len(df)
        print(
            f"Suppression CITY_TO_REMOVE: {removed_count} lignes supprimées sur {total} ({removed_count / total * 100:.2f}%)")

        df = df[~mask_remove].copy()

        # 7) Nettoyage final : strip, remplacer variantes E.Leclerc etc. (tu avais déjà)
        df['osm_name'] = df['osm_name_norm']  # ou garde osm_name si tu préfères l'original non-normalisé
        df['osm_name'] = df['osm_name'].replace({
            'centre commercial e leclerc': 'e.leclerc',
            'e leclerc drive': 'e.leclerc drive',
            'centre commercial e leclerc': 'e.leclerc',
            'e leclerc': 'e.leclerc'
        })

        # Supprimer la colonne temporaire si tu veux
        df = df.drop(columns=['osm_name_norm'], errors='ignore')

        return df

    def filter_eu_eur(self, min_count: int = 20) -> pd.DataFrame:
        """
        Filtre les données pour l'UE et EUR uniquement

        Args:
            min_count: Nombre minimum de produits par pays

        Returns:
            DataFrame filtré
        """
        if self.df_cleaned is None:
            self.clean_data()

        df_eur = self.df_cleaned[
            (self.df_cleaned['currency'] == 'EUR') &
            (self.df_cleaned['location_osm_address_country_code'].isin(self.EU_COUNTRIES))
            ].copy()

        return df_eur

    @staticmethod
    def detect_outliers_iqr(data: pd.Series, factor: float = 1.5) -> pd.Series:
        """
        Détecte les valeurs aberrantes avec la méthode IQR

        Args:
            data: Série de données à analyser
            factor: Facteur multiplicatif pour l'IQR (1.5 par défaut)

        Returns:
            Série booléenne indiquant les outliers
        """
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        return (data < lower_bound) | (data > upper_bound)

    def remove_outliers(self, df: pd.DataFrame, column: str = 'price') -> pd.DataFrame:
        """
        Supprime les outliers d'une colonne

        Args:
            df: DataFrame à nettoyer
            column: Nom de la colonne à traiter

        Returns:
            DataFrame sans outliers
        """
        outliers_mask = self.detect_outliers_iqr(df[column])
        print(
            f"Outliers détectés pour '{column}': {outliers_mask.sum()} sur {len(df)} ({outliers_mask.sum() / len(df) * 100:.2f}%)")

        return df[~outliers_mask].copy()

    def get_country_statistics(self, df: pd.DataFrame, min_count: int = 20) -> pd.DataFrame:
        """
        Calcule les statistiques par pays

        Args:
            df: DataFrame à analyser
            min_count: Nombre minimum de produits par pays

        Returns:
            DataFrame avec les statistiques par pays
        """
        stats = df.groupby('location_osm_address_country')['price'].agg([
            ('moyenne', 'mean'),
            ('médiane', 'median'),
            ('min', 'min'),
            ('max', 'max'),
            ('écart_type', 'std'),
            ('nombre', 'count')
        ]).reset_index()

        stats = stats[stats['nombre'] >= min_count]
        stats = stats.sort_values('moyenne', ascending=False)

        return stats

    def get_store_statistics(self, df: pd.DataFrame, min_count: int = 100) -> pd.DataFrame:
        """
        Calcule les statistiques par magasin/enseigne

        Args:
            df: DataFrame à analyser
            min_count: Nombre minimum de produits par magasin

        Returns:
            DataFrame avec les statistiques par magasin
        """


        stats = df.groupby('osm_name')['price'].agg([
            ('moyenne', 'mean'),
            ('médiane', 'median'),
            ('min', 'min'),
            ('max', 'max'),
            ('écart_type', 'std'),
            ('nombre', 'count')
        ]).reset_index()

        stats = stats[stats['nombre'] >= min_count]
        stats = stats.sort_values('moyenne', ascending=True)

        return stats

    def plot_boxplot_comparison(self, df_before: pd.DataFrame, df_after: pd.DataFrame,
                                column: str = 'price'):
        """
        Compare les distributions avant et après nettoyage

        Args:
            df_before: DataFrame avant nettoyage
            df_after: DataFrame après nettoyage
            column: Colonne à visualiser
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].boxplot(df_before[column].dropna())
        axes[0].set_title(f'{column} - Avant nettoyage')
        axes[0].set_ylabel('Prix (€)')
        axes[0].grid(True, alpha=0.3)

        axes[1].boxplot(df_after[column].dropna())
        axes[1].set_title(f'{column} - Après nettoyage')
        axes[1].set_ylabel('Prix (€)')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("image/plot_boxplot_comparison.png", dpi=300, bbox_inches='tight')

    def plot_mean_prices_by_country(self, df: pd.DataFrame, top_n: int = None):
        """
        Affiche un graphique des prix moyens par pays

        Args:
            df: DataFrame à visualiser
            top_n: Nombre de pays à afficher (tous par défaut)
        """
        mean_prices = (
            df.groupby('location_osm_address_country')['price']
            .mean()
            .sort_values(ascending=True)
        )

        if top_n:
            mean_prices = mean_prices.tail(top_n)

        plt.figure(figsize=(14, 8))
        bars = plt.barh(mean_prices.index, mean_prices.values)

        # Coloration des barres selon le prix
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        plt.xlabel('Prix moyen (€)', fontsize=12)
        plt.ylabel('Pays', fontsize=12)
        plt.title('Prix moyen par pays (EUR) - Données Open Food Facts',
                  fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)

        # Ajout des valeurs sur les barres
        for i, (country, price) in enumerate(mean_prices.items()):
            plt.text(price, i, f' {price:.2f}€',
                     va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig("image/plot_mean_prices_by_country.png", dpi=300, bbox_inches='tight')

    def plot_mean_prices_by_store(self, df: pd.DataFrame, top_n: int = 30):
        """
        Affiche un graphique des prix moyens par magasin/enseigne (du moins cher au plus cher),
        avec le pays entre parenthèses et un dégradé de couleur intuitif.

        Args:
            df: DataFrame à visualiser
            top_n: Nombre de magasins à afficher
        """
        # Calcul du prix moyen par magasin
        mean_prices = (
            df.groupby('osm_name')['price']
            .mean()
            .sort_values(ascending=True)
            .head(top_n)  # les moins chers
        )

        # Récupérer le pays correspondant à chaque magasin
        store_countries = df.groupby('osm_name')['location_osm_address_country'].first()

        # Combiner nom du magasin + pays
        labels = [f"{store} ({store_countries[store]})" for store in mean_prices.index]

        plt.figure(figsize=(14, 10))
        bars = plt.barh(labels, mean_prices.values)

        # Coloration : vert clair (moins cher) → rouge foncé (plus cher)
        cmap = plt.cm.RdYlGn_r  # RdYlGn_r : inversé pour vert=bas, rouge=haut
        norm = plt.Normalize(mean_prices.min(), mean_prices.max())
        colors = [cmap(norm(price)) for price in mean_prices.values]

        for bar, color in zip(bars, colors):
            bar.set_color(color)

        plt.xlabel('Prix moyen (€)', fontsize=12)
        plt.ylabel('Magasin / Enseigne', fontsize=12)
        plt.title(f'Top {top_n} - Magasins les moins chers (EUR)',
                  fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)

        # Ajout des valeurs sur les barres
        for i, price in enumerate(mean_prices.values):
            plt.text(price, i, f' {price:.2f}€', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig("image/plot_mean_prices_by_store.png", dpi=300, bbox_inches='tight')

    def plot_distribution(self, df: pd.DataFrame, column: str = 'price'):
        """
        Affiche la distribution des prix avec histogramme et densité

        Args:
            df: DataFrame à visualiser
            column: Colonne à analyser
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Histogramme
        axes[0].hist(df[column].dropna(), bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Prix (€)')
        axes[0].set_ylabel('Fréquence')
        axes[0].set_title('Distribution des prix')
        axes[0].grid(True, alpha=0.3)

        # Densité (KDE)
        df[column].dropna().plot(kind='density', ax=axes[1], linewidth=2)
        axes[1].set_xlabel('Prix (€)')
        axes[1].set_ylabel('Densité')
        axes[1].set_title('Densité de probabilité des prix')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("image/plot_distribution.png", dpi=300, bbox_inches='tight')


def main():
    """Fonction principale pour exécuter l'analyse"""

    # Initialisation
    analyzer = PriceAnalyzer("dataset/prices.parquet")

    # Nettoyage des données
    print("=" * 60)
    print("NETTOYAGE DES DONNÉES")
    print("=" * 60)
    df_cleaned = analyzer.clean_data()
    print(f"Données chargées : {len(df_cleaned)} lignes")

    # Filtrage UE + EUR
    df_eur = analyzer.filter_eu_eur()
    print(f"Données UE (EUR) : {len(df_eur)} lignes")

    # Statistiques avant nettoyage des outliers
    print("\n" + "=" * 60)
    print("STATISTIQUES PAR PAYS (avant suppression des outliers)")
    print("=" * 60)
    stats_before = analyzer.get_country_statistics(df_eur)
    print(stats_before.to_string(index=False))

    # Visualisation avant/après nettoyage
    print("\n" + "=" * 60)
    print("SUPPRESSION DES OUTLIERS")
    print("=" * 60)
    df_eur_clean = analyzer.remove_outliers(df_eur)

    # Comparaison graphique
    analyzer.plot_boxplot_comparison(df_eur, df_eur_clean)

    # Statistiques après nettoyage
    print("\n" + "=" * 60)
    print("STATISTIQUES PAR PAYS (après suppression des outliers)")
    print("=" * 60)
    stats_after = analyzer.get_country_statistics(df_eur_clean)
    print(stats_after.to_string(index=False))

    # Visualisations finales
    print("\n" + "=" * 60)
    print("VISUALISATIONS")
    print("=" * 60)

    # Distribution des prix
    analyzer.plot_distribution(df_eur_clean)

    # Prix moyens par pays
    analyzer.plot_mean_prices_by_country(df_eur_clean)

    # Statistiques par magasin
    print("\n" + "=" * 60)
    print("STATISTIQUES PAR MAGASIN (top 30)")
    print("=" * 60)

    df_clean_store = analyzer.clean_store(df_eur_clean)
    # Récupérer le pays associé à chaque magasin
    store_countries = df_clean_store.groupby('osm_name')['location_osm_address_country'].first()

    # Calculer les stats par magasin
    store_stats = analyzer.get_store_statistics(df_clean_store, min_count=100)

    # Ajouter le pays entre parenthèses au nom du magasin
    store_stats['osm_name'] = store_stats['osm_name'].apply(lambda x: f"{x} ({store_countries.get(x, '')})")

    # Affichage des 30 premiers
    print(store_stats.to_string(index=False))

    # Prix moyens par magasin
    analyzer.plot_mean_prices_by_store(df_clean_store, top_n=30)


if __name__ == "__main__":
    main()