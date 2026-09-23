"""
preprocessing.py
=================
Etape 1 du projet House Prices : exploration & preparation des donnees.

Ce module regroupe, sous forme de fonctions reutilisables, tout le travail de
nettoyage / encodage / normalisation valide dans le notebook
`01_exploration_preparation.ipynb`. Il est concu pour etre importe :
    - depuis les notebooks suivants (feature engineering, modelisation, ...)
    - depuis l'application Streamlit (pour transformer une nouvelle
      observation exactement comme les donnees d'entrainement)

Regle d'or anti data-leakage : les objets qui "apprennent" des donnees
(imputers, scaler, encoders) sont toujours ajustes (fit) UNIQUEMENT sur le
jeu d'entrainement, jamais sur l'ensemble du dataset avant split. C'est pour
cela que tout le pretraitement "statistique" est encapsule dans un
ColumnTransformer / Pipeline scikit-learn plutot qu'applique "a la main"
sur df entier.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder


# ---------------------------------------------------------------------------
# 1. Constantes : dictionnaires de variables
# ---------------------------------------------------------------------------
# Colonnes techniques a retirer des features (identifiant, cible)
ID_COL = "Id"
TARGET_COL = "SalePrice"

# Colonnes categorielles pour lesquelles NaN signifie "absence de
# l'equipement" et non une valeur manquante au sens statistique.
# On les remplace par la chaine "None".
NONE_FILL_CATEGORICAL = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType",
]

# Colonnes numeriques pour lesquelles NaN signifie "0" (pas d'equipement
# correspondant -> pas de surface).
# NB : GarageYrBlt n'est PAS dans cette liste. Remplacer une annee
# manquante par 0 est incoherent (cela signifierait "garage construit en
# l'an 0"), et fausserait completement toute feature derivee comme l'age
# du garage. Cette colonne recoit un traitement dedie, voir
# fill_garage_year_built() ci-dessous.
ZERO_FILL_NUMERIC = [
    "GarageArea", "GarageCars",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "MasVnrArea",
]

# GarageYrBlt : NaN signifie "pas de garage". La remplacer par 0 est
# incoherent (0 = l'an 0), et une mediane globale gommerait le lien entre
# l'age du garage et celui de la maison. On transforme plutot GarageYrBlt
# en GarageAge = YrSold - GarageYrBlt (age du garage au moment de la
# vente) : c'est une quantite directement interpretable, et l'absence de
# garage devient alors naturellement "age 0" (coherent avec les autres
# zero-fill du groupe garage : GarageArea=0, GarageCars=0). Voir
# create_garage_age() ci-dessous.
GARAGE_YEAR_COL = "GarageYrBlt"
GARAGE_AGE_COL = "GarageAge"
YEAR_SOLD_COL = "YrSold"

# Colonnes avec de vraies valeurs manquantes (peu nombreuses) : imputation
# par le mode (categorielles) ou la mediane (numeriques) - apprise sur train.
MODE_FILL_CATEGORICAL = [
    "MSZoning", "Utilities", "Functional", "Electrical",
    "KitchenQual", "SaleType", "Exterior1st", "Exterior2nd",
]

# LotFrontage est traite a part : imputation par la mediane du quartier
# (Neighborhood), une variable plus informative qu'une mediane globale.
NEIGHBORHOOD_MEDIAN_COL = "LotFrontage"
NEIGHBORHOOD_GROUP_COL = "Neighborhood"

# Colonnes numeriques dont le "vrai type" est categoriel (code, pas quantite)
NUMERIC_TO_CATEGORICAL = ["MSSubClass"]

# Variables categorielles ORDINALES : on encode ces colonnes en respectant
# un ordre logique de qualite plutot qu'en one-hot, pour ne pas perdre
# l'information d'ordre (Po < Fa < TA < Gd < Ex, etc.)
QUALITY_ORDER = ["None", "Po", "Fa", "TA", "Gd", "Ex"]
BSMT_EXPOSURE_ORDER = ["None", "No", "Mn", "Av", "Gd"]
BSMT_FINTYPE_ORDER = ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"]
GARAGE_FINISH_ORDER = ["None", "Unf", "RFn", "Fin"]
FUNCTIONAL_ORDER = ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"]
FENCE_ORDER = ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"]

ORDINAL_FEATURES = {
    "ExterQual": QUALITY_ORDER,
    "ExterCond": QUALITY_ORDER,
    "BsmtQual": QUALITY_ORDER,
    "BsmtCond": QUALITY_ORDER,
    "HeatingQC": QUALITY_ORDER,
    "KitchenQual": QUALITY_ORDER,
    "FireplaceQu": QUALITY_ORDER,
    "GarageQual": QUALITY_ORDER,
    "GarageCond": QUALITY_ORDER,
    "PoolQC": QUALITY_ORDER,
    "BsmtExposure": BSMT_EXPOSURE_ORDER,
    "BsmtFinType1": BSMT_FINTYPE_ORDER,
    "BsmtFinType2": BSMT_FINTYPE_ORDER,
    "GarageFinish": GARAGE_FINISH_ORDER,
    "Functional": FUNCTIONAL_ORDER,
    "Fence": FENCE_ORDER,
}

# Seuil utilise pour le filtrage des valeurs aberrantes (cf. notebook,
# section "incoherences et valeurs aberrantes") : grandes surfaces habitables
# vendues anormalement bas.


# ---------------------------------------------------------------------------
# 2. Chargement et nettoyage de base
# ---------------------------------------------------------------------------
def load_data(path: str | Path) -> pd.DataFrame:
    """Charge le CSV brut."""
    return pd.read_csv(path)


def filter_to_original_rows(df: pd.DataFrame, max_id: int = 1460) -> pd.DataFrame:
    """Ne conserve que les lignes Id <= max_id (1460 par defaut).

    Consigne du formateur : les lignes Id > 1460 posent probleme. Verifie
    empiriquement : sur Id <= 1460, SalePrice suit des valeurs entieres
    "rondes" (vraies transactions, ecart-type ~79 000 $, corr. avec
    OverallQual ~0.79 -- coherent avec le dataset Ames Housing de
    reference). Sur Id > 1460, SalePrice contient des valeurs a
    virgule flottante sur 10+ decimales (ex. 169277.0524984 $), avec un
    ecart-type anormalement faible (~16 500 $) et une corr. quasi nulle
    avec OverallQual (~0.09) : ce ne sont pas de vraies ventes, elles
    diluent le signal si on les garde. A appeler EN PREMIER, juste apres
    load_data() et avant basic_cleaning() (qui retire la colonne Id).
    """
    df = df.copy()
    if ID_COL in df.columns:
        df = df.loc[df[ID_COL] <= max_id]
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage de base :
    - suppression des doublons
    - retrait de la colonne Id (non predictive)
    - correction de type pour MSSubClass (code categoriel, pas numerique)
    """
    df = df.copy()
    df = df.drop_duplicates()

    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    if "MSSubClass" in df.columns:
        df["MSSubClass"] = df["MSSubClass"].astype(str)

    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Retire les observations clairement aberrantes identifiees a
    l'exploration : tres grande surface habitable (GrLivArea > 4000) vendue
    a un prix anormalement bas. Classique sur ce dataset (documente par
    l'auteur original du jeu de donnees).

    A n'appliquer QUE sur le train (jamais sur des donnees a predire).
    """
    df = df.copy()
    if "GrLivArea" in df.columns and TARGET_COL in df.columns:
        mask = (df["GrLivArea"] > 4000) & (
            df[TARGET_COL] < 300000
        )
        df = df.loc[~mask]
    return df


# ---------------------------------------------------------------------------
# 3. Traitement des valeurs manquantes "structurelles" (avant split)
# ---------------------------------------------------------------------------
# Ces remplacements ne "n'apprennent" rien des donnees (pas de moyenne, pas
# de mediane) : ce sont des regles metier fixes, donc on peut les appliquer
# avant le split sans risque de fuite de donnees.
def create_garage_age(df: pd.DataFrame) -> pd.DataFrame:
    """Remplace GarageYrBlt (une annee) par GarageAge (un age en annees),
    beaucoup plus adapte a un remplissage par 0.

    - Garage present (GarageYrBlt renseigne) :
        GarageAge = YrSold - GarageYrBlt
    - Pas de garage (GarageYrBlt = NaN) :
        GarageAge = 0, coherent avec les autres colonnes "absence
        d'equipement" du groupe garage (GarageArea=0, GarageCars=0...) et
        beaucoup plus logique que de mettre 0 dans GarageYrBlt lui-meme
        (qui signifierait "garage construit en l'an 0").
    - Securite : quelques lignes des donnees brutes ont un GarageYrBlt
      posterieur a YrSold (ex. 2207 au lieu de 2007, faute de saisie),
      ce qui donnerait un age negatif -> on borne l'age a 0 (clip).

    Cette transformation ne repose que sur des colonnes de la MEME ligne
    (GarageYrBlt, YrSold), ce n'est pas une statistique calculee sur
    l'ensemble des lignes : elle peut donc etre appliquee avant le split
    train/test sans creer de fuite de donnees.
    """
    df = df.copy()
    if GARAGE_YEAR_COL in df.columns and YEAR_SOLD_COL in df.columns:
        age = df[YEAR_SOLD_COL] - df[GARAGE_YEAR_COL]
        age = age.where(df[GARAGE_YEAR_COL].notna(), 0)   # pas de garage -> age 0
        age = age.clip(lower=0)                            # securite anti-anomalie de saisie
        df[GARAGE_AGE_COL] = age
        df = df.drop(columns=[GARAGE_YEAR_COL])
    return df


def fill_structural_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Remplace les NaN qui signifient une absence reelle d'equipement :
    - "None" pour les categorielles concernees
    - 0 pour les numeriques concernees (surfaces, compteurs)
    - GarageYrBlt est traite a part (voir create_garage_age) car 0 n'a
      aucun sens pour une annee ; il est transforme en GarageAge, qui lui
      peut valoir 0 legitimement.
    """
    df = df.copy()

    for col in NONE_FILL_CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    for col in ZERO_FILL_NUMERIC:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    df = create_garage_age(df)

    return df



# ---------------------------------------------------------------------------
# 3bis. Feature engineering (transformations ligne par ligne, sans fuite)
# ---------------------------------------------------------------------------
# Toutes les variables ci-dessous sont calculees a partir de colonnes de la
# MEME ligne (aucune moyenne/mediane/mode calculee sur l'ensemble des
# lignes) : elles peuvent donc etre ajoutees avant le split train/test,
# exactement comme create_garage_age(). GarageAge n'est PAS recreee ici :
# elle existe deja, produite par create_garage_age() dans
# fill_structural_missing().
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les variables de feature engineering validees pour ce projet.

    - TotalSF        : surface totale habitable (sous-sol + 1er + 2e etage)
    - TotalBath       : nombre equivalent de salles de bain completes
    - HouseAge        : age du logement a la vente (YrSold - YearBuilt)
    - RemodAge        : anciennete depuis la derniere renovation
    - TotalPorchSF    : surface totale des amenagements exterieurs
    - HasPool         : indicateur binaire de presence d'une piscine
    - Has2ndFloor     : indicateur binaire de presence d'un 2e etage
    - OverallScore    : score combine qualite x etat (OverallQual * OverallCond)
    """
    df = df.copy()

    # --- Surfaces agregees ---
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

    df["TotalPorchSF"] = (
        df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"]
        + df["ScreenPorch"] + df["WoodDeckSF"]
    )

    # --- Comptage combine ---
    df["TotalBath"] = (
        df["FullBath"] + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
    )

    # --- Variables temporelles ---
    df["HouseAge"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)
    df["RemodAge"] = (df["YrSold"] - df["YearRemodAdd"]).clip(lower=0)

    # --- Indicateurs binaires (presence/absence) ---
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)

    # --- Score combine ---
    df["OverallScore"] = df["OverallQual"] * df["OverallCond"]

    return df




# ---------------------------------------------------------------------------
# 4. Split X / y puis train / test
# ---------------------------------------------------------------------------
def split_X_y(df: pd.DataFrame, target: str = TARGET_COL):
    """Separe les features (X) de la cible (y)."""
    X = df.drop(columns=target)
    y = df[target]
    return X, y


def train_test_split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.25, random_state: int = 42
):
    """Split train/test. A faire AVANT tout fit d'imputer/scaler/encoder
    pour eviter la fuite de donnees.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# ---------------------------------------------------------------------------
# 5. Listes de features par type (pour le ColumnTransformer)
# ---------------------------------------------------------------------------
def get_feature_lists(X: pd.DataFrame):
    """Repartit les colonnes de X en 3 groupes :
    - numeriques (continues/discretes)
    - categorielles ordinales (ordre de qualite connu)
    - categorielles nominales (pas d'ordre)

    LotFrontage est imputee a part (mediane par quartier), donc retiree ici
    des numeriques "standards" et geree separement dans le pipeline complet.
    """
    ordinal_features = [c for c in ORDINAL_FEATURES if c in X.columns]

    categorical_cols = X.select_dtypes(include=["object", "str", "category"]).columns.tolist()
    nominal_features = [c for c in categorical_cols if c not in ordinal_features]

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

    return numeric_features, ordinal_features, nominal_features


# ---------------------------------------------------------------------------
# 6. Imputation "apprise" (LotFrontage par mediane de quartier + fallback)
# ---------------------------------------------------------------------------
class NeighborhoodMedianImputer:
    """Impute LotFrontage par la mediane de LotFrontage au sein du meme
    Neighborhood. Suit l'API fit/transform de scikit-learn pour s'inserer
    proprement dans un Pipeline, et n'apprend les medianes QUE sur les
    donnees passees a fit() (donc uniquement le train si utilise
    correctement).
    """

    def __init__(self, group_col: str = NEIGHBORHOOD_GROUP_COL,
                 target_col: str = NEIGHBORHOOD_MEDIAN_COL):
        self.group_col = group_col
        self.target_col = target_col
        self.medians_: pd.Series | None = None
        self.global_median_: float | None = None

    def fit(self, X: pd.DataFrame, y=None):
        self.medians_ = X.groupby(self.group_col)[self.target_col].median()
        self.global_median_ = X[self.target_col].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        filled = X[self.target_col].fillna(X[self.group_col].map(self.medians_))
        # Securite : si un quartier inconnu apparait (jamais vu au fit),
        # on retombe sur la mediane globale plutot que de laisser un NaN.
        X[self.target_col] = filled.fillna(self.global_median_)
        return X


def apply_lotfrontage_imputation(X_train, X_test):
    """Applique l'imputation de LotFrontage : fit sur train, transform sur
    train ET test (pas de fuite : les medianes de quartier viennent
    uniquement du train).
    """
    imputer = NeighborhoodMedianImputer()
    imputer.fit(X_train)
    X_train_out = imputer.transform(X_train)
    X_test_out = imputer.transform(X_test)
    return X_train_out, X_test_out, imputer


# ---------------------------------------------------------------------------
# 7. Pipeline scikit-learn (imputation restante + encodage + scaling)
# ---------------------------------------------------------------------------
def build_preprocessing_pipeline(
    numeric_features: list[str],
    ordinal_features: list[str],
    nominal_features: list[str],
) -> ColumnTransformer:
    """Construit le ColumnTransformer applique APRES le split train/test et
    apres fill_structural_missing() + apply_lotfrontage_imputation().

    - numerique      : imputation mediane (pour les quelques NaN residuels,
                        ex. MasVnrArea deja traite mais par securite) +
                        StandardScaler (utile pour les modeles lineaires,
                        neutre pour les modeles a arbres)
    - ordinal        : imputation par le mode + encodage ordinal respectant
                        l'ordre de qualite defini dans ORDINAL_FEATURES
    - nominal        : imputation par le mode + One-Hot Encoding
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    ordinal_categories = [ORDINAL_FEATURES[col] for col in ordinal_features]
    ordinal_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=ordinal_categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    nominal_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("ord", ordinal_pipeline, ordinal_features),
        ("nom", nominal_pipeline, nominal_features),
    ])

    return preprocessor


# ---------------------------------------------------------------------------
# 7bis. Pipeline complet (imputation + encodage + modele) en un seul objet
# ---------------------------------------------------------------------------
def build_full_pipeline(model, X_reference: pd.DataFrame) -> Pipeline:
    """Construit un Pipeline scikit-learn complet, directement fit/predict-
    able sur des donnees "brutes nettoyees" (sorties de
    fill_structural_missing() + add_engineered_features(), donc encore NON
    encodees et NON imputees pour LotFrontage) :

        1) NeighborhoodMedianImputer : impute LotFrontage (mediane de
           quartier, apprise uniquement sur les donnees passees a fit())
        2) ColumnTransformer : imputation residuelle + encodage ordinal/
           one-hot + StandardScaler (cf. build_preprocessing_pipeline)
        3) model : l'estimateur final (regression lineaire, arbre, etc.)

    Interet : un seul objet .fit(X_train, y_train) / .predict(X_new) pour
    toute la chaine, reutilisable tel quel pour l'app Streamlit (etape 8)
    sans avoir a rejouer le pretraitement a la main. X_reference sert
    uniquement a determiner la liste des colonnes numeriques / ordinales /
    nominales (get_feature_lists) ; elle n'est pas utilisee pour fit quoi
    que ce soit ici.
    """
    numeric_features, ordinal_features, nominal_features = get_feature_lists(X_reference)
    preprocessor = build_preprocessing_pipeline(
        numeric_features, ordinal_features, nominal_features
    )
    return Pipeline(steps=[
        ("lotfrontage_imputer", NeighborhoodMedianImputer()),
        ("preprocessor", preprocessor),
        ("model", model),
    ])


# ---------------------------------------------------------------------------
# 8. Fonction "orchestrateur" : reproduit tout le pipeline de l'etape 1
# ---------------------------------------------------------------------------
def run_pipeline(csv_path: str | Path, test_size: float = 0.2,
                        random_state: int = 42, drop_outliers: bool = True):
    """Execute l'integralite de l'etape 1 et renvoie tout ce dont les etapes
    suivantes (feature engineering, modelisation) ont besoin.

    Retour :
        X_train, X_test, y_train, y_test : donnees encodees pretes pour un
            modele (numpy arrays issus du ColumnTransformer)
        preprocessor : ColumnTransformer fitte sur X_train (a reutiliser
            tel quel pour transformer de nouvelles observations, ex. dans
            l'app Streamlit)
        feature_names : noms des colonnes en sortie du preprocessor
    """
    # 1) Chargement + filtrage des lignes non fiables (Id > 1460, cf.
    #    filter_to_original_rows) + nettoyage de base
    df = load_data(csv_path)
    df = filter_to_original_rows(df)
    df = basic_cleaning(df)

    # 2) Valeurs manquantes structurelles (regles fixes, sans fuite possible)
    df = fill_structural_missing(df)

    # 2bis) Feature engineering (transformations ligne par ligne, sans fuite)
    df = add_engineered_features(df)

    # 3) Suppression des outliers evidents (sur l'ensemble avant split, car
    #    ce sont des points clairement invalides pour tout modele, identifies
    #    visuellement lors de l'exploration - pas une statistique apprise)
    if drop_outliers:
        df = remove_outliers(df)

    # 4) Separation X / y
    X, y = split_X_y(df)

    # 5) Split train / test AVANT toute imputation statistique
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, test_size=test_size, random_state=random_state
    )

    # 6) Imputation LotFrontage (apprise sur train uniquement)
    X_train, X_test, _ = apply_lotfrontage_imputation(X_train, X_test)

    # 7) Imputation mode pour les quelques colonnes "vraies manquantes"
    #    a faible cardinalite de NaN -> geree dans le ColumnTransformer
    #    (imputer + encodeur), donc rien a faire ici manuellement.

    # 8) Construction et fit du preprocessor sur le train uniquement
    numeric_features, ordinal_features, nominal_features = get_feature_lists(X_train)
    # LotFrontage deja imputee -> reste dans numeric_features (c'est voulu :
    # le SimpleImputer(median) du pipeline ne fera rien dessus puisqu'il n'y
    # a plus de NaN, il sert de filet de securite pour les autres colonnes).
    preprocessor = build_preprocessing_pipeline(
        numeric_features, ordinal_features, nominal_features
    )

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    return {
        "X_train": X_train_transformed,
        "X_test": X_test_transformed,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "X_train_raw": X_train,
        "X_test_raw": X_test,
    }




if __name__ == "__main__":
    # Petit test manuel : python -m src.preprocessing
    result = run_pipeline("data/raw/House_Prices.csv")
    print("X_train shape:", result["X_train"].shape)
    print("X_test shape :", result["X_test"].shape)
    print("Nb features apres encodage :", len(result["feature_names"]))