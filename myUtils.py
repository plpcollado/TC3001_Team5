# ============================================================
# Archivo: myUtils.py
# Propósito: Funciones utilitarias de limpieza, normalización, 
# preprocesamiento y visualización, útiles para trabajar con 
# el dataset de delitos CDMX.
# ============================================================

import pandas as pd
import numpy as np
import unicodedata
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import os
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime
from collections import Counter
from scipy import sparse
from scipy.sparse import save_npz, load_npz, csr_matrix, hstack


# Para clasificación
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import BallTree
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier  # usado en plots PCA (espacio de clasificación)
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, accuracy_score,
    roc_curve, auc, confusion_matrix, average_precision_score,
    precision_recall_curve, roc_auc_score, brier_score_loss,
    mean_squared_error, mean_absolute_error, r2_score
)

 


# Para clustering especializado
from typing import Tuple, Optional, Sequence, Iterable, Dict, List
from pyproj import CRS, Transformer



from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    pairwise_distances_argmin_min,
)

from sklearn.utils import check_random_state
import joblib
import folium



# Para series de tiempo

import h3
import pygeohash as pgh
from prophet import Prophet
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX



import warnings


# variable global:
_PYPROJ_AVAILABLE = True
R = 500.0  # metros


# ============================================================
# Preprocesamiento
# ============================================================



# Mapeo de meses inglés → español
MESES_MAP = {
    "january": "enero", "february": "febrero", "march": "marzo",
    "april": "abril", "may": "mayo", "june": "junio", "july": "julio",
    "august": "agosto", "september": "septiembre", "october": "octubre",
    "november": "noviembre", "december": "diciembre"
}




# Mapeo de dias inglés → español
DIAS_MAP = {
    "Monday": "lunes",
    "Tuesday": "martes",
    "Wednesday": "miercoles",
    "Thursday": "jueves",
    "Friday": "viernes",
    "Saturday": "sabado",
    "Sunday": "domingo"
}





def apply_dtypes(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Aplica tipos de datos finales al dataset de delitos CDMX (sin usar maps externos).
    - Fechas → datetime64[ns] (parseo robusto con dayfirst=True, errors='coerce')
    - Años → Int64 (nullable)
    - Coordenadas y anio_hecho → Float64 (nullable)
    - Textuales → string (pandas StringDtype)
    
    Imprime un resumen: columnas procesadas, faltantes, y NaT resultantes en fechas.
    Retorna el DataFrame (misma referencia) con dtypes finales.
    """
    # Definición de columnas esperadas por bloque lógico
    date_cols   = ["fecha_inicio", "fecha_hecho"]
    time_cols   = ["hora_inicio", "hora_hecho"]                 # se mantienen como string
    month_cols  = ["mes_inicio", "mes_hecho"]                   # string (normalización aparte)
    int_cols    = ["anio_inicio", "anio_hecho"]                 # años con faltantes → Int64
    float_cols  = ["latitud", "longitud"]         # Float64 (nullable)
    text_cols   = [
        "delito", "categoria_delito", "competencia", "fiscalia",
        "agencia", "unidad_investigacion", "colonia_hecho", "colonia_catalogo",
        "alcaldia_hecho", "alcaldia_catalogo", "municipio_hecho"
    ]

    expected = set(date_cols + time_cols + month_cols + int_cols + float_cols + text_cols)

    # 1) Reporte de columnas faltantes/sobrantes
    missing = [c for c in expected if c not in df.columns]
    extras  = [c for c in df.columns if c not in expected]
    if verbose:
        print(f"Columnas esperadas totales: {len(expected)}")
        if missing:
            print(f"Faltan {len(missing)} columnas: {missing}")
        else:
            print("No faltan columnas esperadas.")
        if extras:
            print(f"Columnas adicionales presentes (no tipificadas): {len(extras)} → {extras}")

    # 2) Conversión de tipos (solo columnas existentes)
    # Textuales (incluye meses y horas): usar pandas StringDtype
    to_string = [c for c in (text_cols + time_cols + month_cols) if c in df.columns]
    for c in to_string:
        try:
            df[c] = df[c].astype("string")
        except Exception as e:
            if verbose:
                print(f"No se pudo convertir '{c}' a string: {e}")

    # Años (enteros con nulos permitidos)
    for c in [x for x in int_cols if x in df.columns]:
        try:
            # Primero asegurar numérico; luego a Int64 (nullable)
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        except Exception as e:
            if verbose:
                print(f"No se pudo convertir '{c}' a Int64: {e}")

    # Float64 (anio_hecho, latitud, longitud)
    for c in [x for x in float_cols if x in df.columns]:
        try:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Float64")
        except Exception as e:
            if verbose:
                print(f"No se pudo convertir '{c}' a Float64: {e}")

    # 3) Parseo de fechas a datetime
    nat_report = {}
    for c in [x for x in date_cols if x in df.columns]:
        # Contar no-nulos “visibles” antes del parseo (para entender NaT introducidos)
        before_non_null = df[c].notna().sum()
        try:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
            after_nat = df[c].isna().sum()
            nat_report[c] = {"no_nulos_antes": int(before_non_null), "NaT_despues": int(after_nat)}
        except Exception as e:
            if verbose:
                print(f"No se pudo parsear '{c}' a datetime: {e}")

    # 4) Impresión para seguimiento
    if verbose:
        print("\nResumen de casting:")
        if to_string:
            print(f"• A string: {len(to_string)} columnas → {to_string}")
        if int_cols:
            present = [c for c in int_cols if c in df.columns]
            if present:
                print(f"• A Int64: {present}")
        if float_cols:
            present = [c for c in float_cols if c in df.columns]
            if present:
                print(f"• A Float64: {present}")
        if date_cols:
            present = [c for c in date_cols if c in df.columns]
            if present:
                print(f"• A datetime64[ns]: {present}")
                if nat_report:
                    for k, v in nat_report.items():
                        print(f"   - '{k}': no nulos antes = {v['no_nulos_antes']}, NaT después = {v['NaT_despues']}")

    return df





# Normalización general de texto
def normalize_text(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Normaliza el texto de una columna específica:
    - Convierte a minúsculas
    - Elimina espacios extra
    - Quita tildes y caracteres especiales
    - Devuelve el DataFrame modificado

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame que contiene la columna a limpiar.
    col : str
        Nombre de la columna a normalizar.

    Retorna
    -------
    pd.DataFrame
        DataFrame con la columna normalizada.
    """
    if col not in df.columns:
        raise ValueError(f"La columna '{col}' no existe en el DataFrame.")

    # Convertimos a string y limpiamos espacios
    df[col] = (
        df[col]
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .fillna("")  # por seguridad
    )

    # Quitamos tildes y caracteres especiales
    df[col] = df[col].apply(
        lambda x: unicodedata.normalize("NFKD", x)
        .encode("ascii", errors="ignore")
        .decode("utf-8")
        if isinstance(x, str) else x
    )

    return df




# Normalización de meses (español + inglés)
def normalize_months(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Normaliza nombres de meses (maneja español/inglés y errores comunes)."""
    df[col] = (
        df[col]
        .astype(str)
        .str.lower()
        .str.strip()
        .replace(MESES_MAP)
        .str.replace(r"^abr$", "abril", regex=True)
        .str.replace(r"^jun$", "junio", regex=True)
        .str.replace(r"^jul$", "julio", regex=True)
        .str.replace(r"^ago$", "agosto", regex=True)
        .str.replace(r"^sept?$", "septiembre", regex=True)
        .str.replace(r"^dic$", "diciembre", regex=True)
        .str.capitalize()
    )
    return df







# Eliminación de nulos críticos en columnas críticas
def drop_missing_rows(df: pd.DataFrame, cols_required: list) -> pd.DataFrame:
    """
    Elimina filas con valores nulos (NaN o None) en las columnas críticas especificadas.
    Imprime cuántas filas fueron eliminadas.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    cols_required : list
        Lista de columnas que no deben contener valores nulos.

    Retorna
    -------
    pd.DataFrame
        DataFrame sin filas con valores nulos en las columnas críticas.
    """
    for col in cols_required:
        if col not in df.columns:
            raise ValueError(f"La columna requerida '{col}' no existe en el DataFrame.")

    total_inicial = len(df)
    df = df.dropna(subset=cols_required)
    total_final = len(df)
    eliminadas = total_inicial - total_final

    print(f"Filas eliminadas por nulos en columnas críticas: {eliminadas} de {total_inicial} ({(eliminadas/total_inicial)*100:.2f}%)")

    return df






# Conversión de meses en texto a número (1-12)
def map_month_to_number(df: pd.DataFrame, col: str = "mes_inicio") -> pd.DataFrame:
    """
    Convierte los nombres de meses (en español o inglés) a su valor numérico (1–12)
    y sobrescribe la columna original.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame que contiene la columna con los meses.
    col : str, opcional
        Nombre de la columna a convertir. Por defecto 'mes_inicio'.

    Retorna
    -------
    pd.DataFrame
        Mismo DataFrame con la columna convertida a valores numéricos (Int64).
    """

    # Diccionario robusto: español e inglés, minúsculas
    month_map = {
        "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
        "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9,
        "octubre": 10, "noviembre": 11, "diciembre": 12,
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
    }

    # Validar existencia
    if col not in df.columns:
        raise ValueError(f"La columna '{col}' no existe en el DataFrame.")

    # Normalizar texto y mapear
    df[col] = (
        df[col]
        .astype(str)
        .str.lower()
        .str.strip()
        .map(month_map)
        .astype("Int64")  # Int64 permite nulos (<NA>)
    )

    # Reporte
    n_total = len(df)
    n_missing = df[col].isna().sum()
    print(f"Columna '{col}' convertida a valores numéricos (1–12).")
    print(f" Registros sin mapeo: {n_missing} de {n_total} ({n_missing/n_total:.2%})")

    return df






# ============================================================
# Feature Engineering
# ============================================================

def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrae características temporales a partir de las columnas
    'fecha_inicio' y 'hora_inicio'.
    
    Variables generadas:
    - anio_inicio (ya existente, asegurada como int)
    - mes_inicio_num (1-12)
    - dia_semana (0=lunes ... 6=domingo)
    - hora (0-23)
    - tipo_turno (mañana/tarde/noche)
    - fin_de_semana (bool)
    - trimestre (1-4)
    - estacionalidad: sin/cos de mes y hora (transformaciones cíclicas)
    """

    # 1. Asegurar que fecha y hora estén en formato datetime 
    df["fecha_inicio"] = pd.to_datetime(df["fecha_inicio"], errors="coerce", dayfirst=True)
    
    # Procesar hora si está como string tipo '12:30:00'
    df["hora_inicio"] = pd.to_datetime(df["hora_inicio"], format="%H:%M:%S", errors="coerce").dt.hour

    # 2. Variables básicas temporales 
    df["anio_inicio"] = df["anio_inicio"].astype("Int64", errors="ignore")

    df["mes_inicio_num"] = map_month_to_number(df, col="mes_inicio")["mes_inicio"].astype("Int64")


    df["dia_semana"] = df["fecha_inicio"].dt.dayofweek.astype("Int64")   # 0=Lunes, 6=Domingo
    df["hora"] = df["hora_inicio"].astype("float").fillna(0).astype(int)
    df["trimestre"] = ((df["mes_inicio_num"] - 1) // 3 + 1).astype("Int64")

    # 3. Tipo de turno (mañana, tarde, noche) 
    def get_turno(hora):
        if pd.isna(hora):
            return "desconocido"
        elif 6 <= hora < 12:
            return "mañana"
        elif 12 <= hora < 19:
            return "tarde"
        else:
            return "noche"

    df["tipo_turno"] = df["hora"].apply(get_turno)

    # 4. Fin de semana 
    df["fin_de_semana"] = df["dia_semana"].isin([5, 6])  # Sábado=5, Domingo=6

    # 5. Transformaciones cíclicas (estacionalidad)
    # Mes
    df["mes_sin"] = np.sin(2 * np.pi * df["mes_inicio_num"] / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes_inicio_num"] / 12)

    # Hora
    df["hora_sin"] = np.sin(2 * np.pi * df["hora"] / 24)
    df["hora_cos"] = np.cos(2 * np.pi * df["hora"] / 24)

    return df






def calc_distancia_centro(df: pd.DataFrame,
                          lat_col: str = "latitud",
                          lon_col: str = "longitud",
                          lat_centro: float = 19.432608,
                          lon_centro: float = -99.133209) -> pd.DataFrame:
    """
    Calcula la distancia (en km) desde cada punto al centro de CDMX.
    Usa la fórmula del haversine.
    """

    R = 6371  # radio de la Tierra (km)

    lat1 = np.radians(df[lat_col])
    lon1 = np.radians(df[lon_col])
    lat2 = radians(lat_centro)
    lon2 = radians(lon_centro)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    df["distancia_centro_cdmx"] = R * c
    return df






def add_geo_clusters(df: pd.DataFrame,
                     lat_col: str = "latitud",
                     lon_col: str = "longitud",
                     n_clusters: int = 20,
                     random_state: int = 42) -> pd.DataFrame:
    """
    Asigna cada punto a un cluster geográfico usando KMeans (técnica de
    clustering) sobre latitud/longitud.
    """

    valid_mask = df[lat_col].notna() & df[lon_col].notna()
    coords = df.loc[valid_mask, [lat_col, lon_col]]

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df.loc[valid_mask, "cluster_geo"] = kmeans.fit_predict(coords)

    df["cluster_geo"] = df["cluster_geo"].fillna(-1).astype("Int64")
    print(f"  Clusters geográficos asignados (KMeans con {n_clusters} grupos).")
    return df







def compute_densidad_delictiva(df: pd.DataFrame,
                               lat_col: str = "latitud",
                               lon_col: str = "longitud",
                               radio_km: float = 1.0,
                               include_self: bool = True) -> pd.DataFrame:
    """
    Densidad delictiva = número de eventos dentro de un radio (km) alrededor de cada punto.
    Usa BallTree + métrica haversine (coordenadas en radianes) y cálculo vectorizado.

    Params
    ------
    include_self : si True, cuenta al propio punto; si False, resta 1 al conteo.
    """

    # Asegurar numérico y máscara válida
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    valid_mask = df[lat_col].notna() & df[lon_col].notna()

    # Si no hay suficientes puntos válidos, devolver ceros
    if valid_mask.sum() == 0:
        df["densidad_delictiva_1km"] = 0
        return df

    # Coordenadas en radianes como matriz NumPy (n, 2) de floats
    coords_deg = df.loc[valid_mask, [lat_col, lon_col]].to_numpy(dtype=float)
    coords_rad = np.radians(coords_deg)

    # BallTree en haversine; radio en radianes
    tree = BallTree(coords_rad, metric="haversine")
    radio = radio_km / 6371.0

    # Conteo vectorizado de vecinos dentro del radio
    counts_valid = tree.query_radius(coords_rad, r=radio, count_only=True)

    # Excluir el propio punto 
    if not include_self:
        counts_valid = counts_valid - 1
        counts_valid[counts_valid < 0] = 0

    # Colocar resultados a la serie completa
    counts = np.zeros(len(df), dtype=np.int64)
    counts[valid_mask.to_numpy()] = counts_valid
    df["densidad_delictiva_1km"] = counts

    print(f"  Densidad delictiva calculada en radio ≈ {radio_km} km (include_self={include_self}).")
    return df







def extract_spatial_features(df: pd.DataFrame,
                             n_clusters: int = 20,
                             radio_km: float = 1.0) -> pd.DataFrame:
    """
    Construye las features espaciales clave:
    - latitud, longitud (ya en df)
    - alcaldia_hecho (categoría)
    - cluster_geo (KMeans)
    - densidad_delictiva_1km (BallTree)
    - distancia_centro_cdmx (float)
    """

    # Convertir a numérico por seguridad
    df["latitud"] = pd.to_numeric(df["latitud"], errors="coerce")
    df["longitud"] = pd.to_numeric(df["longitud"], errors="coerce")

    # Ejecutar cálculos
    df = calc_distancia_centro(df)
    df = add_geo_clusters(df, n_clusters=n_clusters)
    df = compute_densidad_delictiva(df, radio_km=radio_km)

    # Asegurar tipos
    df["alcaldia_hecho"] = df["alcaldia_hecho"].astype("string")

    return df









def fill_missing_competencia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa valores faltantes en la columna 'competencia' con 'desconocido'.
    """
    if "competencia" not in df.columns:
        raise ValueError("No se encontró la columna 'competencia' en el DataFrame.")
    n_missing = df["competencia"].isna().sum()
    df["competencia"] = df["competencia"].fillna("desconocido")
    print(f" Competencia imputada con 'desconocido' en {n_missing} registros.")
    return df






def frequency_encode(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Aplica codificación por frecuencia (frequency encoding) a una columna categórica.
    Retorna una serie numérica normalizada (0-1).
    """
    freq = df[col].value_counts(normalize=True)
    encoded = df[col].map(freq)
    print(f" Frequency encoding aplicado a '{col}' ({len(freq)} categorías).")
    return encoded




def target_encode(df: pd.DataFrame, col: str, target_col: str) -> pd.Series:
    """
    Aplica codificación target (media de frecuencia del target por categoría).
    Ideal para modelos supervisados.

    target_col debe ser categórico (ej: 'categoria_delito').
    """
    target_mean = df.groupby(col)[target_col].value_counts(normalize=True).unstack(fill_value=0)
    # Si hay una categoría dominante, podemos reducir dimensionalidad tomando la más frecuente:
    major = df[target_col].value_counts(normalize=True).idxmax()
    enc = df[col].map(target_mean[major])  # mapea probabilidad del delito más común
    print(f" Target encoding aplicado a '{col}' usando '{target_col}' (clase base: {major}).")
    return enc





def conditional_dummies(df: pd.DataFrame, col: str, threshold: int = 20) -> pd.DataFrame:
    """
    Si la columna tiene < threshold categorías únicas, aplica one-hot encoding.
    """
    n_cat = df[col].nunique()
    if n_cat <= threshold:
        df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=False)
        print(f" Dummies creados para '{col}' ({n_cat} categorías).")
    else:
        print(f" '{col}' tiene {n_cat} categorías → se mantiene codificación numérica.")
    return df





def extract_institutional_features(df: pd.DataFrame, target_col: str = "categoria_delito") -> pd.DataFrame:
    """
    Construye el bloque de contexto institucional:
    - Imputa competencia
    - Aplica codificación (freq/target)
    - Genera dummies si corresponde
    """

    df = fill_missing_competencia(df)

    cat_cols = ["fiscalia", "agencia", "unidad_investigacion", "competencia"]

    for col in cat_cols:
        if df[col].nunique() < 20:
            df = conditional_dummies(df, col)
        else:
            df[f"{col}_freq"] = frequency_encode(df, col)
            df[f"{col}_target"] = target_encode(df, col, target_col)

    return df








# =======================
# TF-IDF sobre columna texto
# =======================

def _prep_text_series(df, text_col: str, normalize_fn=None):
    """
    Prepara la serie de texto: string dtype, fillna, strip.
    Si pasas normalize_fn (p.ej. normalize_text), la aplica.
    """
    s = df[text_col].astype("string").fillna("").astype(str).str.strip()
    if normalize_fn is not None:
        # normalize_fn debe aceptar (df, col) o una serie; detectamos firma simple
        try:
            tmp = s.to_frame(name=text_col)
            tmp = normalize_fn(tmp, text_col)
            s = tmp[text_col]
        except Exception:
            # fallback si es una función que opera sobre Series
            s = normalize_fn(s)
    return s



def fit_tfidf_on_train(df_train,
                       text_col: str = "delito",
                       ngram_range=(1, 2),
                       max_features: int = 20000,
                       min_df=2,
                       max_df=0.95,
                       lowercase=False,
                       use_idf=True,
                       sublinear_tf=True,
                       dtype=np.float32,
                       normalize_fn=None,
                       stop_words=None):
    """
    Ajusta un TfidfVectorizer sobre df_train[text_col] y devuelve (vectorizer, X_train_tfidf).
    Se recomienda pasar normalize_fn = normalize_text para limpiar acentos y espacios.
    """
    vec = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        lowercase=lowercase,   # ya normalizamos fuera si queremos
        use_idf=use_idf,
        sublinear_tf=sublinear_tf,
        dtype=dtype,
        stop_words=stop_words
    )
    s_train = _prep_text_series(df_train, text_col, normalize_fn=normalize_fn)
    X_train = vec.fit_transform(s_train)
    print(f" TF-IDF ajustado sobre '{text_col}': {X_train.shape[0]} docs × {X_train.shape[1]} términos")
    return vec, X_train



def transform_tfidf(df,
                    vectorizer: TfidfVectorizer,
                    text_col: str = "delito",
                    normalize_fn=None):
    """
    Transforma df[text_col] con un vectorizer ya ajustado.
    """
    s = _prep_text_series(df, text_col, normalize_fn=normalize_fn)
    X = vectorizer.transform(s)
    print(f"  TF-IDF transform: {X.shape[0]} docs × {X.shape[1]} términos")
    return X








def month_cyclic_diff(m1: pd.Series, m2: pd.Series) -> pd.Series:
    """
    Diferencia cíclica mínima entre meses (1..12).
    Ej.: mes_hecho=12, mes_inicio=1 -> diff_ciclica = 1 (no -11)
    Retorna valores en [-6, +6].
    """
    diff = (m1 - m2) % 12
    # Reubicar en rango [-6, +6]
    diff = diff.where(diff <= 6, diff - 12)
    return diff.astype("Int64")



# ------------------------------------------------------------
# Lag features temporales entre fecha_inicio y fecha_hecho
# Requiere:
#   - fecha_inicio, fecha_hecho (parseables a datetime, dayfirst)
#   - mes_inicio_num (1..12). Si no existe, lo crea desde mes_inicio.
#   - mes_hecho_num (1..12). Lo crea desde mes_hecho si es texto.
# Usa: map_month_to_number(df, col)
# ------------------------------------------------------------
def extract_coherence_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea variables de coherencia temporal:
    - retraso_denuncia_dias = (fecha_inicio - fecha_hecho) en días (Float64)
    - diferencia_mes = mes_inicio_num - mes_hecho_num (Int64, puede ser negativa)
    - diferencia_mes_ciclica = diff cíclica mínima [-6, +6] (Int64)
    - año_diferente = (año de fecha_inicio != año de fecha_hecho) (bool)

    Notas:
    - Maneja NaT con errors='coerce' y dayfirst=True.
    - Si faltan 'mes_inicio_num' o 'mes_hecho_num', los genera desde 'mes_inicio'/'mes_hecho'.
    - Imprime resumen de nulos/valores negativos en el retraso.
    """

    # 1) Parseo robusto de fechas
    df["fecha_inicio"] = pd.to_datetime(df.get("fecha_inicio"), errors="coerce", dayfirst=True)
    df["fecha_hecho"]  = pd.to_datetime(df.get("fecha_hecho"),  errors="coerce", dayfirst=True)

    # 2) Asegurar meses en formato numérico (1..12)
    if "mes_inicio_num" not in df.columns:
        if "mes_inicio" not in df.columns:
            raise ValueError("Falta 'mes_inicio_num' y no existe 'mes_inicio' para mapear.")
        df["mes_inicio_num"] = map_month_to_number(df.copy(), col="mes_inicio")["mes_inicio"].astype("Int64")

    if "mes_hecho_num" not in df.columns:
        if "mes_hecho" not in df.columns:
            raise ValueError("Falta 'mes_hecho_num' y no existe 'mes_hecho' para mapear.")
        df["mes_hecho_num"] = map_month_to_number(df.copy(), col="mes_hecho")["mes_hecho"].astype("Int64")

    # 3) Retraso en días (Float64)
    #    (fecha_inicio - fecha_hecho).dt.days puede ser negativo si hay incoherencias
    delta = (df["fecha_inicio"] - df["fecha_hecho"]).dt.total_seconds() / (24 * 3600)
    df["retraso_denuncia_dias"] = pd.to_numeric(delta, errors="coerce").astype("Float64")

    # 4) Diferencias de mes
    df["diferencia_mes"] = (df["mes_inicio_num"] - df["mes_hecho_num"]).astype("Int64")
    df["diferencia_mes_ciclica"] = month_cyclic_diff(df["mes_inicio_num"], df["mes_hecho_num"])

    # 5) Años diferentes
    anio_ini = df["fecha_inicio"].dt.year
    anio_hec = df["fecha_hecho"].dt.year
    df["año_diferente"] = (anio_ini != anio_hec) & anio_ini.notna() & anio_hec.notna()

    # 6) Resumen útil
    n_total = len(df)
    n_nat = df["retraso_denuncia_dias"].isna().sum()
    n_neg = (df["retraso_denuncia_dias"] < 0).sum()
    print(" Coherencia temporal (lag) – resumen")
    print(f"  Registros totales: {n_total:,}")
    print(f"  retraso_denuncia_dias NaN: {n_nat:,} ({n_nat/n_total:.2%})")
    print(f"  retraso_denuncia_dias negativos: {n_neg:,} ({n_neg/n_total:.2%})")

    return df




def save_tfidf_outputs(X_train, X_test, vectorizer, base_dir: str, prefix: str = "tfidf_delito"):
    """
    Guarda matrices sparse .npz y nombres de features .csv en BASE_DIR/data/
    """
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    train_path = os.path.join(data_dir, f"{prefix}_train.npz")
    test_path  = os.path.join(data_dir, f"{prefix}_test.npz")
    vocab_path = os.path.join(data_dir, f"{prefix}_features.csv")

    save_npz(train_path, X_train)
    save_npz(test_path, X_test)

    feats = vectorizer.get_feature_names_out()
    pd.Series(feats, name="feature").to_csv(vocab_path, index=False, encoding="utf-8-sig")

    print(" Guardado TF-IDF:")
    print(f"   - Train: {train_path}")
    print(f"   - Test : {test_path}")
    print(f"   - Vocab: {vocab_path}")





def save_features(df: pd.DataFrame, feature_cols: list, base_dir: str, filename: str = "features_temporales.csv"):
    """
    Guarda las columnas especificadas en un archivo CSV dentro del
    directorio comun de datos. Crea el directorio si no existe.
    """
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    path = os.path.join(data_dir, filename)
    df[feature_cols].to_csv(path, index=False, encoding="utf-8-sig")

    print(f"Features guardadas en: {path}")








def sparse_overview(X, name="X"):
    """
    Muestra un resumen tipo info() para una matriz sparse.
    """
    if not sparse.isspmatrix(X):
        raise TypeError("Se espera una matriz scipy.sparse")

    X = X.tocsr()  # asegura CSR
    n_rows, n_cols = X.shape
    nnz = X.nnz
    density = nnz / (n_rows * n_cols)
    dtype = X.dtype

    # Memoria aproximada (datos + índices). No incluye Python overhead.
    mem_bytes = X.data.nbytes + X.indices.nbytes + X.indptr.nbytes
    mem_mb = mem_bytes / (1024**2)

    print(f"\n{name}: CSR matrix")
    print(f"  shape        : {n_rows:,} x {n_cols:,}")
    print(f"  nnz          : {nnz:,} (densidad: {density:.6%})")
    print(f"  dtype        : {dtype}")
    print(f"  approx memory: {mem_mb:.2f} MB")
    # distribución básica de no-ceros por fila (muestra)
    sample = min(5, n_rows)
    nz_per_row = np.diff(X.indptr)
    print(f"  nz/fila (p10, p50, p90): {np.percentile(nz_per_row, [10,50,90]).astype(int)}")








# Análisis del desbalance de clases
def analyze_class_distribution(y_train: pd.Series, y_test: pd.Series, plot: bool = True):
    """
    Analiza la distribución de clases en y_train / y_test y sugiere si aplicar rebalanceo.
    """
    print(" Distribución de clases en TRAIN:")
    dist_train = y_train.value_counts(normalize=True).sort_values(ascending=False)
    print(dist_train.to_frame("Proporción").applymap(lambda x: f"{x:.3%}"))

    print("\n Distribución de clases en TEST:")
    dist_test = y_test.value_counts(normalize=True).sort_values(ascending=False)
    print(dist_test.to_frame("Proporción").applymap(lambda x: f"{x:.3%}"))

    # Ratio entre clase mayoritaria y minoritaria
    ratio = dist_train.max() / dist_train.min()
    print(f"\n  Ratio (mayor/minor): {ratio:.1f}x")

    # Sugerencia
    if ratio > 5:
        print("  Dataset fuertemente desbalanceado: usar técnicas de rebalanceo (class_weight, SMOTE, etc.)")
    elif ratio > 2:
        print("  Desbalance moderado: considerar pesos por clase o F1 macro.")
    else:
        print("  Distribución relativamente balanceada: no se requiere rebalanceo explícito.")

    # Visualización
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        dist_train.plot(kind="bar", ax=ax[0], color="steelblue", title="Distribución TRAIN")
        dist_test.plot(kind="bar", ax=ax[1], color="orange", title="Distribución TEST")
        for a in ax: a.set_ylabel("Proporción"); a.grid(axis="y", linestyle="--", alpha=0.6)
        plt.suptitle("Distribución de clases (train/test)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.show()

    return dist_train, dist_test, ratio






#  Pesos por clase (para modelos sklearn con class_weight='balanced')
#    y sample_weight por instancia (útil en XGBoost y otros)
def make_class_weights(y: pd.Series):
    """
    Calcula pesos balanceados por clase y vector de sample_weight.
    Limpia automáticamente NaN, espacios y capitalización.
    """
    # Validación básica
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    # Limpieza de etiquetas (quita espacios, NaN y homogeneiza mayúsculas/minúsculas)
    y = (
        y.astype(str)
        .str.strip()
        .str.lower()
        .replace({"nan": np.nan})
        .dropna()
    )

    classes = np.unique(y)

    # Calcular pesos balanceados
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weight_dict = {cls: w for cls, w in zip(classes, cw)}

    # Generar sample_weight
    weight_map = pd.Series(class_weight_dict)
    sample_weight = y.map(weight_map).to_numpy(dtype=np.float32)

    print(f"Clases detectadas: {len(classes)} → {list(classes)}")
    return class_weight_dict, sample_weight





#  Pesos derivados para XGBoost multiclase (opcional)
#  En multiclase NO hay scale_pos_weight directo por clase; se usan sample_weight.
#  Aun así, puedes tener un mapeo inverso de frecuencia como referencia.
def make_inverse_freq_weights(y: pd.Series):
    counts = y.value_counts()
    inv = 1.0 / counts
    inv = inv / inv.mean()  # normaliza alrededor de 1.0
    return inv.to_dict()








# ============================================================
# OOF, evaluación CV y visualizaciones de modelos de clasificación
# ============================================================


def _supports_predict_proba(est) -> bool:
    return hasattr(est, "predict_proba")

def _supports_decision_function(est) -> bool:
    return hasattr(est, "decision_function")

def _is_binary(scores: np.ndarray) -> bool:
    return scores.ndim == 1 or (scores.ndim == 2 and scores.shape[1] == 1)

def _scores_to_2col(scores_1d: np.ndarray) -> np.ndarray:
    """Convierte scores 1D (binario) a 2 columnas [1-p, p] para ROC macro."""
    if scores_1d.ndim != 1:
        scores_1d = scores_1d.ravel()
    # si es decision function (puede no estar en [0,1]), lo dejamos como “score” relativo
    # para ROC OvR funciona; si fueran probabilidades, mejor aún.
    # Normalizamos a [0,1] solo si es claramente probabilidad fuera de rango? — no forzamos.
    s = scores_1d.astype(float)
    # construir “pseudo-proba” de la clase positiva:
    # si ya es probabilidad, perfecto; si es margen, solo reescala la comparación (ROC es rank-based).
    # Para estabilidad en trazado, usamos sigmoide suave en valores muy grandes
    # pero evitamos tocar si ya parece proba en [0,1].
    if s.min() < 0 or s.max() > 1:
        s = 1 / (1 + np.exp(-s))  # sigmoide
    return np.vstack([1.0 - s, s]).T


# ----------------------------
# ROC macro a partir de scores OvR
# ----------------------------
def macro_roc_auc_from_scores(y_true, y_score) -> float:
    """
    AUC-ROC macro vía One-vs-Rest con interpolación en grilla común.
    y_score: matriz (n_samples, n_clases) con probabilidades o decision_function.
    Si es binario con vector 1D, conviértelo primero con _scores_to_2col.
    """
    y_true = np.asarray(y_true)
    labels = np.unique(y_true)
    if y_score is None:
        return np.nan

    # Binarización multiclase
    Y = label_binarize(y_true, classes=labels)  # (n, C)
    if Y.ndim == 1:  # corner para binario
        Y = np.c_[1 - Y, Y]

    fpr_grid = np.linspace(0, 1, 500)
    tpr_accum = np.zeros_like(fpr_grid)

    for j in range(Y.shape[1]):
        fpr_c, tpr_c, _ = roc_curve(Y[:, j], y_score[:, j])
        # interpola en grilla común
        tpr_interp = np.interp(fpr_grid, fpr_c, tpr_c)
        tpr_interp[0] = 0.0
        tpr_accum += tpr_interp

    tpr_macro = tpr_accum / Y.shape[1]
    return auc(fpr_grid, tpr_macro)



def _decode_oof_labels(oof_dict_enc, label_encoder):
    """Convierte y_true/y_pred ints → strings; 
    preserva y_score y el orden de clases."""
    oof_dec = {}
    classes_txt = list(label_encoder.classes_)  # orden consistente con y_score columnas

    for name, rec in oof_dict_enc.items():
        y_true_enc = np.asarray(rec["y_true"])
        y_pred_enc = np.asarray(rec["y_pred"])
        y_score    = rec["y_score"]  # (n, K) ya en el orden del encoder

        # Decodificar a texto
        y_true_txt = label_encoder.inverse_transform(y_true_enc)
        y_pred_txt = label_encoder.inverse_transform(y_pred_enc)

        # Guardar nueva entrada con etiquetas de texto
        oof_dec[name] = {
            "y_true": y_true_txt,
            "y_pred": y_pred_txt,
            "y_score": y_score,      # columnas ya alineadas a classes_txt
            "classes": np.array(classes_txt)
        }
    return oof_dec






# ----------------------------
# OOF por modelo (pred, scores)
# ----------------------------
def collect_oof_predictions_old(models, X, y, cv, verbose=True):
    """
    Genera predicciones OOF (out-of-fold) para cada (name, estimator) en 'models'.
    - y_pred OOF para Macro-F1, Balanced Acc, Accuracy.
    - y_score OOF (predict_proba o decision_function) para ROC AUC macro.
    Maneja binario/multiclase y sparse/dense. Si falla (p.ej. modelo no soporta sparse),
    reporta y continúa sin romper el flujo.
    Devuelve:
      oof_dict[name] = {"y_true","y_pred","y_score","classes"}
      df_summary      = DataFrame con macro_f1_oof, bal_acc_oof, acc_oof, roc_auc_macro_oof
    """
    classes_sorted = np.unique(y)
    oof = []
    rows = []

    for name, est in models:
        if verbose:
            print(f"OOF → {name}")
        try:
            # 1) Etiquetas OOF (si el modelo falla con sparse, caerá al except)
            y_pred = cross_val_predict(est, X, y, cv=cv, n_jobs=-1, method="predict")

            # 2) Scores OOF (para ROC)
            y_score = None
            try:
                if _supports_predict_proba(est):
                    y_score = cross_val_predict(est, X, y, cv=cv, n_jobs=-1, method="predict_proba")
                elif _supports_decision_function(est):
                    y_score = cross_val_predict(est, X, y, cv=cv, n_jobs=-1, method="decision_function")
                    # Ajuste binario 1D
                    if _is_binary(y_score):
                        y_score = _scores_to_2col(np.array(y_score).ravel())
            except Exception as e_sc:
                if verbose:
                    print(f"   (sin scores para {name}: {e_sc})")

            macro_f1 = f1_score(y, y_pred, average="macro")
            bal_acc  = balanced_accuracy_score(y, y_pred)
            acc      = accuracy_score(y, y_pred)

            roc_auc_macro = np.nan
            if y_score is not None:
                try:
                    roc_auc_macro = macro_roc_auc_from_scores(y, y_score)
                except Exception as e_auc:
                    if verbose:
                        print(f"   (AUC macro no calculable para {name}: {e_auc})")

            oof.append({
                "model": name,
                "y_true": y.copy(),
                "y_pred": y_pred,
                "y_score": y_score,
                "classes": classes_sorted
            })
            rows.append({
                "model": name,
                "macro_f1_oof": macro_f1,
                "bal_acc_oof": bal_acc,
                "acc_oof": acc,
                "roc_auc_macro_oof": roc_auc_macro
            })

        except Exception as e_model:
            if verbose:
                print(f"   {name} falló en OOF: {e_model}")
            # aún así dejar registro vacío para mantener tabla consistente
            rows.append({
                "model": name,
                "macro_f1_oof": np.nan,
                "bal_acc_oof": np.nan,
                "acc_oof": np.nan,
                "roc_auc_macro_oof": np.nan
            })

    df_summary = pd.DataFrame(rows).sort_values(
        by=["macro_f1_oof","bal_acc_oof"], ascending=False
    ).reset_index(drop=True)

    return {d["model"]: d for d in oof}, df_summary



# FIT en TRAIN para uso correcto de MLP ===
def make_dense_for_mlp(X_tfidf_csr, X_tab_df, n_svd=400, seed=42):
    # 1) SVD sobre TF-IDF (solo TRAIN)
    svd = TruncatedSVD(n_components=n_svd, random_state=seed)
    X_tfidf_svd = svd.fit_transform(X_tfidf_csr)  # (n, n_svd) denso

    # 2) Tabular a denso
    X_tab_dense = X_tab_df.values.astype(np.float32, copy=False)

    # 3) Concat y escalar
    X_dense = np.hstack([X_tfidf_svd.astype(np.float32), X_tab_dense])
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_dense_scaled = scaler.fit_transform(X_dense).astype(np.float32)

    return X_dense_scaled, svd, scaler





# TRANSFORM en TEST para MLP (usar svd/scaler ajustados en TRAIN) 
def transform_dense_for_mlp(X_tfidf_csr_test, X_tab_df_test, svd, scaler):
    X_tfidf_svd_test = svd.transform(X_tfidf_csr_test)
    X_tab_dense_test = X_tab_df_test.values.astype(np.float32, copy=False)
    X_dense_test = np.hstack([X_tfidf_svd_test.astype(np.float32), X_tab_dense_test])
    return scaler.transform(X_dense_test).astype(np.float32)











def to_numeric_sparse(df: pd.DataFrame, fillna=0.0, verbose=True) -> pd.DataFrame:
    """
    Deja el DF listo para csr_matrix:
      - Selecciona solo columnas numéricas/booleanas
      - bool -> uint8
      - rellena NaN con 'fillna'
      - castea a float32
    """
    num_like = df.select_dtypes(include=["number", "bool"]).copy()
    if verbose:
        dropped = [c for c in df.columns if c not in num_like.columns]
        if dropped:
            print(f" Columnas no numéricas descartadas ({len(dropped)}): {dropped[:10]}{' ...' if len(dropped)>10 else ''}")
    # bool -> uint8
    for c in num_like.select_dtypes(include=["bool"]).columns:
        num_like[c] = num_like[c].astype(np.uint8)
    # NaN + float32
    if fillna is not None:
        num_like = num_like.fillna(fillna)
    return num_like.astype(np.float32)
















def collect_oof_predictions(models, X, y, cv, fit_params=None, verbose=True):
    """
    Genera predicciones OOF (out-of-fold) para cada (name, estimator) en 'models'.
    - y_pred OOF para Macro-F1, Balanced Acc, Accuracy.
    - y_score OOF (predict_proba o decision_function) para ROC AUC macro.
    Maneja binario/multiclase y sparse/dense. Si falla (p.ej. modelo no soporta sparse),
    reporta y continúa sin romper el flujo.

    Requiere que existan en el módulo:
      - _supports_predict_proba(est)
      - _supports_decision_function(est)
      - _is_binary(scores)
      - _scores_to_2col(scores_1d)
      - macro_roc_auc_from_scores(y_true, y_score)

    Parámetros
    ----------
    models : list[(str, estimator)]
        Lista de (nombre, estimador sklearn-compatible).
    X : array-like o scipy.sparse
        Matriz de características (solo TRAIN).
    y : array-like o pd.Series
        Etiquetas de entrenamiento.
    cv : objeto de validación cruzada (e.g., StratifiedKFold).
    fit_params : dict | None
        Parámetros para pasar a .fit() dentro de cross_val_predict (p.ej. {"sample_weight": vector}).
    verbose : bool
        Si True, imprime el progreso y avisos.

    Retorna
    -------
    oof_dict : dict[name] -> {"y_true","y_pred","y_score","classes"}
    df_summary : pd.DataFrame con columnas:
        [model, macro_f1_oof, bal_acc_oof, acc_oof, roc_auc_macro_oof]
        ordenado por Macro-F1 (desc) y Balanced Acc (desc).
    """
    if fit_params is None:
        fit_params = {}

    classes_sorted = np.unique(y)
    oof = []
    rows = []

    for name, est in models:
        if verbose:
            print(f"OOF → {name}")
        try:
            # 1) Etiquetas OOF
            y_pred = cross_val_predict(
                est, X, y, cv=cv, n_jobs=-1, method="predict", fit_params=fit_params
            )

            # 2) Scores OOF (para ROC/AUC)
            y_score = None
            try:
                if _supports_predict_proba(est):
                    y_score = cross_val_predict(
                        est, X, y, cv=cv, n_jobs=-1, method="predict_proba", fit_params=fit_params
                    )
                elif _supports_decision_function(est):
                    raw_scores = cross_val_predict(
                        est, X, y, cv=cv, n_jobs=-1, method="decision_function", fit_params=fit_params
                    )
                    # Ajuste binario 1D → 2 columnas
                    if _is_binary(raw_scores):
                        y_score = _scores_to_2col(np.array(raw_scores).ravel())
                    else:
                        y_score = np.asarray(raw_scores)
            except Exception as e_sc:
                if verbose:
                    print(f"   (sin scores para {name}: {e_sc})")

            # 3) Métricas OOF
            macro_f1 = f1_score(y, y_pred, average="macro")
            bal_acc  = balanced_accuracy_score(y, y_pred)
            acc      = accuracy_score(y, y_pred)

            roc_auc_macro = np.nan
            if y_score is not None:
                try:
                    roc_auc_macro = macro_roc_auc_from_scores(y, y_score)
                except Exception as e_auc:
                    if verbose:
                        print(f"   (AUC macro no calculable para {name}: {e_auc})")

            # 4) Acumular resultados
            oof.append({
                "model": name,
                "y_true": y.copy(),
                "y_pred": y_pred,
                "y_score": y_score,
                "classes": classes_sorted
            })
            rows.append({
                "model": name,
                "macro_f1_oof": macro_f1,
                "bal_acc_oof": bal_acc,
                "acc_oof": acc,
                "roc_auc_macro_oof": roc_auc_macro
            })

        except Exception as e_model:
            if verbose:
                print(f"    {name} falló en OOF: {e_model}")
            rows.append({
                "model": name,
                "macro_f1_oof": np.nan,
                "bal_acc_oof": np.nan,
                "acc_oof": np.nan,
                "roc_auc_macro_oof": np.nan
            })

    df_summary = (
        pd.DataFrame(rows)
        .sort_values(by=["macro_f1_oof","bal_acc_oof"], ascending=False)
        .reset_index(drop=True)
    )

    return {d["model"]: d for d in oof}, df_summary










# ----------------------------
# Boxplot Macro-F1 por modelo (CV)
# ----------------------------
def boxplot_macro_f1_cv(models, X, y, cv,
                        annotate=True,
                        max_xticks=30,
                        print_summary=True):
    """
    Recolecta Macro-F1 fold-a-fold para cada modelo y traza boxplot + jitter.
    Devuelve (df_scores, summary).
    """
    all_scores = []
    for name, est in models:
        try:
            scores = cross_val_score(est, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
            for i, s in enumerate(scores, start=1):
                all_scores.append({"model": name, "fold": i, "macro_f1": s})
        except Exception as e:
            # si el modelo no soporta sparse u otra cosa, se salta pero se informa
            print(f"    {name} sin CV-score: {e}")

    df_scores = pd.DataFrame(all_scores)
    if df_scores.empty:
        print("No se recolectaron puntajes. Revisa compatibilidad de modelos con X.")
        return df_scores, pd.DataFrame()

    summary = (
        df_scores.groupby("model")["macro_f1"]
        .agg(mean="mean", std="std", median="median",
             q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
    )
    summary["iqr"] = summary["q75"] - summary["q25"]
    summary = summary.sort_values(by="median", ascending=False)
    order_by_median = summary.index.tolist()

    plt.figure(figsize=(14, max(6, 0.35*len(order_by_median))))
    ax = sns.boxplot(data=df_scores, x="model", y="macro_f1",
                     order=order_by_median, showfliers=False, linewidth=1)
    sns.stripplot(data=df_scores, x="model", y="macro_f1",
                  order=order_by_median, alpha=0.55, jitter=0.18, size=4)

    if annotate:
        med_vals = summary["median"].values
        for xtick, med in enumerate(med_vals):
            ax.text(xtick, med + 0.004, f"{med:.3f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Macro-F1 (CV=5)")
    ax.set_xlabel("Modelos")
    ax.set_title("Desempeño por modelo (CV) — Boxplot + puntos", fontsize=16, weight="bold")
    ax.set_ylim(-0.02, 1.02)
    rot = 90 if len(order_by_median) > max_xticks else 45
    plt.xticks(rotation=rot)
    plt.tight_layout()
    plt.show()

    if print_summary:
        top_n = min(10, len(summary))
        print("\n Top-10 por mediana de Macro-F1:")
        display(summary.head(top_n)[["median", "mean", "std", "iqr"]].round(3))

        print("\n Top-10 más estables (menor std):")
        display(summary.sort_values("std").head(top_n)[["median", "mean", "std", "iqr"]].round(3))

    return df_scores, summary









# ----------------------------
# Confusion matrices para Top-N (OOF)
# ----------------------------
def plot_confusions_topN(oof_dict, df_oof, N=10, normalize=True, fmt_decimals=2):
    """
    Matrices de confusión OOF para los Top-N por macro_f1_oof (desempate por bal_acc).
    """
    topN = df_oof.sort_values(by=["macro_f1_oof","bal_acc_oof"], ascending=False).head(N)
    if topN.empty:
        print("No hay modelos con OOF para graficar.")
        return

    model_names = topN["model"].tolist()

    # Tomamos el conjunto de clases de la primera entrada
    first = oof_dict.get(model_names[0], None)
    if first is None:
        print("No se encontraron oof para los modelos seleccionados.")
        return
    labels_full = np.array(sorted(np.unique(first["y_true"])))
    codes = np.array([f"L{i+1}" for i in range(len(labels_full))])

    n = len(model_names)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5*rows))
    axes = np.array(axes).reshape(-1)

    for ax, name in zip(axes, model_names):
        oo = oof_dict[name]
        y_true, y_pred = np.asarray(oo["y_true"]), np.asarray(oo["y_pred"])
        cm = confusion_matrix(
            y_true, y_pred, labels=labels_full,
            normalize="true" if normalize else None
        )
        # para anotaciones limpias
        if normalize:
            fmt = f".{fmt_decimals}f"
            annot_data = np.where(np.isnan(cm), 0.0, cm)
        else:
            fmt = "d"
            annot_data = cm.astype(int)

        sns.heatmap(
            annot_data, ax=ax, cmap="Blues", cbar=True,
            xticklabels=codes, yticklabels=codes,
            annot=True, fmt=fmt, annot_kws={"fontsize":8}
        )
        ax.set_title(name, fontsize=11, pad=8)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.tick_params(axis="x", rotation=0, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)

    for k in range(n, rows*cols):
        fig.delaxes(axes[k])

    # Leyenda global al pie
    plt.suptitle(f"Matrices de confusión OOF — Top-{N} (Lk = etiqueta)", fontsize=14, y=0.99)
    plt.tight_layout()
    # leyenda textual:
    legend_text = "   ".join([f"{c}: {l}" for c, l in zip(codes, labels_full)])
    plt.figtext(0.5, 0.01, legend_text, ha="center", fontsize=9, family="monospace")
    plt.show()









# ----------------------------
# ROC macro Top-N (OOF)
# ----------------------------
def plot_macro_roc_topN(oof_dict, df_oof, N=10):
    """
    Traza ROC macro OOF de los Top-N modelos (por roc_auc_macro_oof).
    Solo para modelos con y_score disponible.
    """
    df_scores = df_oof[~df_oof["roc_auc_macro_oof"].isna()].copy()
    if df_scores.empty:
        print("Ningún modelo tiene scores OOF; no es posible trazar ROC.")
        return

    df_top = df_scores.sort_values(by="roc_auc_macro_oof", ascending=False).head(N)

    markers = ['.', ',', 'o', 'v', '^', '>', '<', '*', '1', '2', '3', '4', '8', 's', 'p', 'P', 'h', 'H', '+', 'x', '|', '_']*3
    lines   = ['-', '--', '-.', ':']*15

    plt.figure(figsize=(12, 8))
    for i, row in enumerate(df_top.itertuples(index=False)):
        name = row.model
        oo = oof_dict[name]
        y_true, y_score = np.asarray(oo["y_true"]), oo["y_score"]
        if y_score is None:
            continue
        auc_macro = macro_roc_auc_from_scores(y_true, y_score)
        fpr_grid = np.linspace(0, 1, 500)
        # reconstrucción de curva macro para trazo (misma que en cálculo)
        labels = np.unique(y_true)
        Y = label_binarize(y_true, classes=labels)
        if Y.ndim == 1:
            Y = np.c_[1 - Y, Y]
        tpr_accum = np.zeros_like(fpr_grid)
        for j in range(Y.shape[1]):
            fpr_c, tpr_c, _ = roc_curve(Y[:, j], y_score[:, j])
            tpr_interp = np.interp(fpr_grid, fpr_c, tpr_c)
            tpr_interp[0] = 0.0
            tpr_accum += tpr_interp
        tpr_macro = tpr_accum / Y.shape[1]

        plt.plot(fpr_grid, tpr_macro,
                 label=f"{name} (AUC={auc_macro:.3f})",
                 marker=markers[i], markevery=40, linestyle=lines[i])

    plt.plot([0,1],[0,1], 'r-', linewidth=1)
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC macro — Top-{N} modelos (OOF)", fontsize=18, weight="bold")
    plt.legend(fontsize=9, loc="lower right", frameon=True)
    plt.tight_layout()
    plt.show()






# ----------------------------
# Espacio de predicción (PCA 2D de scores OOF)
# ----------------------------
# ============================================================
# Paleta y marcadores consistentes por nº de clases
# ============================================================
def _get_palette_and_markers(n_classes: int):
    """
    Devuelve (colors, markers) con longitud n_classes.
    - Colores: 'tab20' repetido si hace falta.
    - Marcadores: conjunto variado, se repite si n_classes > len(base).
    """
    import seaborn as sns
    base_colors = sns.color_palette("tab20", n_colors=max(n_classes, 10)).as_hex()
    colors = base_colors[:n_classes]
    base_markers = ['o', 's', '^', 'v', 'P', 'X', 'D', '*', '<', '>', 'h', 'H']
    markers = (base_markers * ((n_classes // len(base_markers)) + 1))[:n_classes]
    return colors, markers






# ============================================================
# Espacio de predicción OOF — PCA(2D) + Regiones coloreadas
# ============================================================
def plot_prediction_space_topN(
    oof_dict, df_oof, N=5,
    sample_size=2000, random_state=42,
    # mejoras:
    order_by=("macro_f1_oof","bal_acc_oof"),
    knn_k=15, grid_step=300,
    region_alpha=0.35, boundary_alpha=0.65
):
    """
    Visualiza, para los Top-N modelos (ordenados por 'order_by'):
      1) PCA(2D) de y_score (predict_proba/decision_function) OOF,
      2) Regiones de decisión coloreadas (KNN en el plano PCA),
      3) Nube de puntos reales encima,
      4) Leyenda global compacta.

    Parámetros
    ----------
    oof_dict : dict[name] -> {'y_true','y_pred','y_score','classes'}
    df_oof   : DataFrame con columnas ['model','macro_f1_oof','bal_acc_oof','roc_auc_macro_oof',...]
    N        : nº de modelos a mostrar
    sample_size : nº máx. de puntos a graficar por modelo
    random_state: semilla reproducible
    order_by    : tupla de columnas para ordenar el Top-N
    knn_k       : vecinos para KNN que define las regiones
    grid_step   : resolución de la malla (más alto = más fino)
    region_alpha: transparencia del relleno de regiones
    boundary_alpha: transparencia de las líneas de frontera
    """

    # Filtra a modelos con y_score disponible y elige Top-N
    df_scores = (df_oof[~df_oof["roc_auc_macro_oof"].isna()]
                 .sort_values(by=list(order_by), ascending=False))
    df_top = df_scores.head(N).copy()
    if df_top.empty:
        print("No hay modelos con scores OOF para visualizar.")
        return

    model_names = df_top["model"].tolist()

    # Clases (usamos las del primer modelo para etiquetas/leyenda)
    first = oof_dict[model_names[0]]
    labels_full = np.array(sorted(np.unique(first["y_true"])))
    n_classes = len(labels_full)
    colors, markers = _get_palette_and_markers(n_classes)
    cmap = ListedColormap(colors)

    # Layout
    cols = 2
    rows = int(np.ceil(len(model_names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.8*cols, 5.4*rows))
    axes = np.array(axes).reshape(-1)

    rng = np.random.default_rng(random_state)

    for ax, name in zip(axes, model_names):
        oo = oof_dict[name]
        y_true = np.asarray(oo["y_true"])
        y_score = oo["y_score"]

        if y_score is None:
            ax.set_title(f"{name} (sin scores)", fontsize=11)
            ax.axis("off")
            continue

        y_score = np.asarray(y_score)
        n = y_true.shape[0]
        m = min(sample_size, n)
        idx = rng.choice(n, size=m, replace=False)

        y_true_sub  = y_true[idx]
        y_score_sub = y_score[idx, :]

        # PCA(2D) sobre los scores
        pca = PCA(n_components=2, random_state=random_state)
        X_pca = pca.fit_transform(y_score_sub)

        # Codificar clases a enteros para colorear y ajustar KNN
        y_codes, class_labels = pd.factorize(y_true_sub)
        nC = len(class_labels)

        # KNN para definir regiones de decisión en el plano PCA
        knn = KNeighborsClassifier(n_neighbors=knn_k)
        knn.fit(X_pca, y_codes)

        # Malla regular (con padding) para pintar regiones
        pad = 0.08
        x_min, x_max = X_pca[:, 0].min(), X_pca[:, 0].max()
        y_min, y_max = X_pca[:, 1].min(), X_pca[:, 1].max()
        x_pad = (x_max - x_min) * pad or 1e-6
        y_pad = (y_max - y_min) * pad or 1e-6

        xx, yy = np.meshgrid(
            np.linspace(x_min - x_pad, x_max + x_pad, grid_step),
            np.linspace(y_min - y_pad, y_max + y_pad, grid_step)
        )
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        # RELLENO DE REGIONES (colores por región)
        ax.contourf(xx, yy, Z, levels=np.arange(-0.5, nC+0.5, 1),
                    cmap=cmap, alpha=region_alpha, antialiased=True)
        # LÍNEAS DE FRONTERA
        ax.contour(xx, yy, Z, levels=np.arange(-0.5, nC+0.5, 1),
                   colors="k", linewidths=0.6, alpha=boundary_alpha)

        # Nube de puntos encima
        for k in range(nC):
            sel = (y_codes == k)
            ax.scatter(
                X_pca[sel, 0], X_pca[sel, 1],
                s=18, alpha=0.85, edgecolor='k', linewidth=0.35,
                c=[colors[k % len(colors)]], marker=markers[k % len(markers)],
                label=str(class_labels[k])
            )

        # Título con métricas
        row = df_top[df_top["model"] == name].iloc[0]
        mF1 = row.get("macro_f1_oof", np.nan)
        aucm = row.get("roc_auc_macro_oof", np.nan)
        ax.set_title(f"{name}\nMacro-F1 OOF={mF1:.3f} | AUC={aucm:.3f}", fontsize=11)
        ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 2")

    # Ocultar ejes sobrantes si N impar
    for ax in axes[len(model_names):]:
        ax.axis("off")

    # Título general
    fig.suptitle(f"Espacios de Predicción OOF — Top-{N} (PCA 2D + Regiones)", fontsize=16, y=0.99)

    # Leyenda global al pie
    handles, labels = axes[0].get_legend_handles_labels() if len(model_names) else ([], [])
    if labels:
        ncol = 3 if n_classes > 8 else 2
        fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=True, fontsize=9,
                   title="Clases", bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout()
    plt.show()




                        
# Usar modelos entrenados
def smoke_test_models(models_dict, label_encoder=None, topk=3):
    """
    Para cada modelo:
      - Usa n_features_in_ para crear un vector CSR de ceros de la forma correcta.
      - Llama predict (y predict_proba si existe) y muestra el Top-k.
    """
    from numpy import argsort

    results = []
    for name, model in models_dict.items():
        nfeat = getattr(model, "n_features_in_", None)
        if nfeat is None:
            print(f"⚠  {name}: no expone n_features_in_; omito.")
            continue

        # 1xN vector de ceros (float32) en CSR
        x_dummy = csr_matrix((1, nfeat), dtype=np.float32)

        try:
            # Si hay probabilidades, sacamos Top-k
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(x_dummy)[0]
                idx = argsort(probs)[::-1][:topk]
                classes = getattr(model, "classes_", None)
                if classes is None:
                    labels = idx
                else:
                    # Decodifica si tenemos label_encoder y las clases son numéricas
                    if label_encoder is not None and np.issubdtype(np.array(classes).dtype, np.integer):
                        labels = label_encoder.inverse_transform(classes[idx])
                    else:
                        labels = classes[idx]
                print(f"\n{name}\n  n_features_in_={nfeat}\n  Top-{topk}:")
                for r, (lab, p) in enumerate(zip(labels, probs[idx]), 1):
                    print(f"   {r}. {lab}  (p={p:.3f})")
                results.append((name, labels[0], probs[idx][0]))

            else:
                # Sin proba: usamos predict directo
                pred = model.predict(x_dummy)[0]
                if label_encoder is not None and isinstance(pred, (np.integer, int)):
                    pred = label_encoder.inverse_transform([pred])[0]
                print(f"\n{name}\n  n_features_in_={nfeat}\n  Predicción: {pred}")
                results.append((name, pred, None))

        except Exception as e:
            print(f" {name} falló en smoke test: {e}")

    return results










# ======================================================
#  INFERENCIA: Construcción de features y predicción XGB
# ======================================================



# ------------------------------------------------------
# Generar bloque temporal (idéntico a test)
# ------------------------------------------------------
def build_temporal_features(df):
    df = df.copy()
    # Asegurar tipo fecha
    for col in ["fecha_hecho", "fecha_inicio"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Año, mes, día de la semana
    df["anio_inicio"] = df["fecha_inicio"].dt.year
    df["mes_inicio_num"] = df["fecha_inicio"].dt.month
    df["dia_semana"] = df["fecha_inicio"].dt.dayofweek

    # Fin de semana
    df["fin_de_semana"] = df["dia_semana"].isin([5, 6]).astype(int)

    # Hora numérica
    def hora_to_num(h):
        try:
            t = datetime.strptime(h.strip(), "%H:%M")
            return t.hour + t.minute / 60.0
        except Exception:
            return np.nan
    df["hora_num"] = df["hora_inicio"].apply(hora_to_num)

    # Codificación trigonométrica
    df["hora_sin"] = np.sin(2 * np.pi * df["hora_num"] / 24)
    df["hora_cos"] = np.cos(2 * np.pi * df["hora_num"] / 24)
    df["mes_sin"] = np.sin(2 * np.pi * df["mes_inicio_num"] / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes_inicio_num"] / 12)
    return df




# ------------------------------------------------------
# Construcción del bloque tabular (sin escalar)
# ------------------------------------------------------
def build_tabular_block(df_raw, artifacts):
    df = build_temporal_features(df_raw)

    # Variables numéricas relevantes
    num_cols = [
        "anio_inicio", "mes_inicio_num", "dia_semana", "hora_num",
        "hora_sin", "hora_cos", "mes_sin", "mes_cos", "latitud", "longitud"
    ]
    df_tab = df[num_cols].fillna(0).astype(np.float32)

    # Alinear columnas con las usadas en entrenamiento
    if artifacts["tabcols"] is not None:
        train_cols = artifacts["tabcols"]
        for c in train_cols:
            if c not in df_tab.columns:
                df_tab[c] = 0
        df_tab = df_tab[train_cols]
    else:
        print(" tabular_columns.json no encontrado; se usará el orden actual.")

    X_tab_csr = csr_matrix(df_tab.values, dtype=np.float32)
    return X_tab_csr


# ------------------------------------------------------
# Construcción del bloque TF-IDF (texto “delito”)
# ------------------------------------------------------
def build_tfidf_block(df_raw, artifacts):
    """
    Construye el bloque TF-IDF sobre la columna 'delito' replicando el pipeline de test.
    Reutiliza normalize_text(df, col) (versión DataFrame+columna) en lugar de aplicar
    una función elemento-a-elemento.
    """
    vectorizer = artifacts.get("tfidf")
    if vectorizer is None:
        raise RuntimeError("Vectorizer TF-IDF no encontrado en artifacts['tfidf'].")

    if "delito" not in df_raw.columns:
        raise ValueError("El DataFrame debe tener la columna 'delito' para TF-IDF.")

    # Reusar tu función de normalización que opera sobre DataFrame+col
    tmp = df_raw[["delito"]].copy()
    normalize_text(tmp, "delito")  # <- tu función existente

    # Asegurar string y sin NaNs
    text_norm = tmp["delito"].fillna("").astype(str)
    X_tfidf = vectorizer.transform(text_norm)
    return X_tfidf


# ------------------------------------------------------
# Predicción con modelo XGBoost (final)
# ------------------------------------------------------
def predict_xgb_from_raw(df_raw, artifacts, topk=5):
    """
    Dado un DataFrame 'df_raw' con las columnas originales humanas,
    aplica las mismas transformaciones de test y predice con XGBoost.
    """
    #artifacts = load_inference_artifacts(base_dir)
    xgb_model = artifacts["xgb"]
    encoder = artifacts["encoder"]

    # Bloques
    X_tab = build_tabular_block(df_raw, artifacts)
    X_tfidf = build_tfidf_block(df_raw, artifacts)

    # Unir (CSR)
    X_full = hstack([X_tab, X_tfidf], format="csr")
    print("Feature vector de la instancia a predecir:")
    print(X_full)
    print(f"Shape final: {X_full.shape}")

    # Verificar consistencia de features
    nfeat_model = getattr(xgb_model, "n_features_in_", None)
    if nfeat_model != X_full.shape[1]:
        raise ValueError(f"Dimensión inconsistente: modelo espera {nfeat_model}, matriz tiene {X_full.shape[1]}")

    # Predicciones
    probs = xgb_model.predict_proba(X_full)
    classes = encoder.inverse_transform(np.arange(probs.shape[1]))

    results = []
    for i in range(len(df_raw)):
        row = probs[i]
        order = np.argsort(row)[::-1][:topk]
        top_preds = [(classes[j], float(row[j])) for j in order]
        results.append({"caso": i, "topk": top_preds})
    return results









# =========================================
# PARA GEO CLUSTERING ESPECIALIZADO
# =========================================


def assert_required_columns(
    df: pd.DataFrame,
    required: Sequence[str],
    context: str = "dataset"
) -> None:
    """
    Verifica que el DataFrame contenga todas las columnas requeridas.
    Lanza AssertionError con mensaje amigable si falta alguna.

    Parameters
    ----------
    df : DataFrame
    required : lista/tupla de columnas requeridas
    context : descripción corta para el mensaje de error
    """
    missing = [c for c in required if c not in df.columns]
    assert len(missing) == 0, (
        f"[{context}] Faltan columnas requeridas: {missing}. "
        f"Columnas disponibles: {list(df.columns)}"
    )


def validate_coordinates(
    df: pd.DataFrame,
    lat_col: str = "latitud",
    lon_col: str = "longitud",
    allow_nulls: bool = False
) -> Dict[str, int]:
    """
    Valida rangos básicos de coordenadas y reporta conteos de problemas.

    Returns
    -------
    dict con:
      - null_lat
      - null_lon
      - out_of_range
    """
    out = {"null_lat": 0, "null_lon": 0, "out_of_range": 0}
    if not allow_nulls:
        out["null_lat"] = int(df[lat_col].isna().sum())
        out["null_lon"] = int(df[lon_col].isna().sum())
        if out["null_lat"] > 0 or out["null_lon"] > 0:
            warnings.warn(
                f"Se encontraron nulos en {lat_col}={out['null_lat']}, "
                f"{lon_col}={out['null_lon']}. Considera depurar."
            )

    mask_out = (
        (df[lat_col].notna() & ((df[lat_col] < -90) | (df[lat_col] > 90))) |
        (df[lon_col].notna() & ((df[lon_col] < -180) | (df[lon_col] > 180)))
    )
    out["out_of_range"] = int(mask_out.sum())
    if out["out_of_range"] > 0:
        warnings.warn(f"Se encontraron {out['out_of_range']} filas con coordenadas fuera de rango.")

    return out


# -----------------------------
# CRS local y proyección
# -----------------------------
def get_local_crs(
    center_lat: float,
    center_lon: float,
    method: str = "aeqd"
) -> str:
    """
    Devuelve un CRS local (proj4) centrado en (center_lat, center_lon).
    method:
      - "aeqd": Azimuthal Equidistant (bueno para distancias radiales)
      - "laea": Lambert Azimuthal Equal-Area (áreas)
      - "lcc" : Lambert Conformal Conic (regional, latitudes medias)

    Nota:
      Elegimos 'aeqd' por simplicidad para distancias euclídeas locales.
    """
    method = method.lower()
    assert method in {"aeqd", "laea", "lcc"}, "Método CRS no soportado."
    if method == "aeqd":
        proj4 = f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    elif method == "laea":
        proj4 = f"+proj=laea +lat_0={center_lat} +lon_0={center_lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    else:  # lcc
        # Dos paralelos estándar razonables alrededor de la latitud central
        lat1 = center_lat - 2.0
        lat2 = center_lat + 2.0
        proj4 = (
            f"+proj=lcc +lat_1={lat1} +lat_2={lat2} "
            f"+lat_0={center_lat} +lon_0={center_lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        )
    return proj4


def project_to_local_meters(
    df: pd.DataFrame,
    lat_col: str = "latitud",
    lon_col: str = "longitud",
    center_lat: float = 19.4326,
    center_lon: float = -99.1332,
    method: str = "aeqd",
    x_name: str = "x_m",
    y_name: str = "y_m",
    inplace: bool = False
) -> pd.DataFrame:
    """
    Proyecta columnas de lat/lon (WGS84) a coordenadas en metros en un CRS local.

    Parameters
    ----------
    df : DataFrame con lat_col, lon_col
    center_lat, center_lon : centro del CRS local
    method : 'aeqd' | 'laea' | 'lcc'
    x_name, y_name : nombres de columnas a crear
    inplace : si True, modifica df; si False, retorna copia

    Returns
    -------
    DataFrame con columnas [x_name, y_name] en metros.

    Raises
    ------
    ImportError si pyproj no está disponible.
    AssertionError si faltan columnas o hay tipos incompatibles.
    """
    assert_required_columns(df, [lat_col, lon_col], context="project_to_local_meters")

    if not _PYPROJ_AVAILABLE:
        raise ImportError(
            "pyproj no está disponible. Instala con `pip install pyproj` para usar proyección local."
        )

    crs_src = CRS.from_epsg(4326)  # WGS84
    crs_dst = CRS.from_proj4(get_local_crs(center_lat, center_lon, method=method))
    transformer = Transformer.from_crs(crs_src, crs_dst, always_xy=True)

    # pyproj espera lon, lat
    xs, ys = transformer.transform(df[lon_col].values, df[lat_col].values)

    if inplace:
        df[x_name] = xs
        df[y_name] = ys
        return df
    else:
        out = df.copy()
        out[x_name] = xs
        out[y_name] = ys
        return out


# -----------------------------
# División temporal Holdout
# -----------------------------
def temporal_holdout_split(
    df: pd.DataFrame,
    date_col: str,
    cutoff: pd.Timestamp,
    include_cutoff_in: str = "test"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa el DataFrame en train/test con base en una fecha de corte.

    Parameters
    ----------
    date_col : nombre de columna con fecha (datetime64)
    cutoff : fecha de corte (pd.Timestamp)
    include_cutoff_in : 'train' o 'test' para decidir dónde cae la igualdad

    Returns
    -------
    (df_train, df_test)
    """
    assert_required_columns(df, [date_col], context="temporal_holdout_split")
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        raise TypeError(f"{date_col} debe ser datetime64[ns]. Convierte antes de llamar.")

    if include_cutoff_in == "train":
        df_train = df[df[date_col] <= cutoff].copy()
        df_test  = df[df[date_col] >  cutoff].copy()
    else:
        df_train = df[df[date_col] <  cutoff].copy()
        df_test  = df[df[date_col] >= cutoff].copy()

    return df_train, df_test







def _safe_silhouette(X: np.ndarray, labels: np.ndarray, random_state: Optional[int] = None, sample_size: Optional[int] = None) -> float:
    """
    Calcula Silhouette de forma segura, con submuestreo opcional para acelerar.
    Devuelve np.nan si no se puede calcular (p.ej. un único cluster o un cluster vacío).
    """
    unique = np.unique(labels)
    if unique.size < 2 or unique.size > X.shape[0] - 1:
        return float("nan")

    if sample_size is not None and X.shape[0] > sample_size:
        rng = check_random_state(random_state)
        idx = rng.choice(X.shape[0], size=sample_size, replace=False)
        X_eval = X[idx]
        y_eval = labels[idx]
    else:
        X_eval = X
        y_eval = labels

    try:
        return float(silhouette_score(X_eval, y_eval, metric="euclidean"))
    except Exception:
        return float("nan")


def _safe_davies_bouldin(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Calcula Davies–Bouldin; devuelve np.nan si no se puede calcular.
    """
    unique = np.unique(labels)
    if unique.size < 2:
        return float("nan")
    try:
        return float(davies_bouldin_score(X, labels))
    except Exception:
        return float("nan")


def _safe_calinski_harabasz(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Calcula Calinski–Harabasz; devuelve np.nan si no se puede calcular.
    """
    unique = np.unique(labels)
    if unique.size < 2:
        return float("nan")
    try:
        return float(calinski_harabasz_score(X, labels))
    except Exception:
        return float("nan")


def compute_internal_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    random_state: Optional[int] = None,
    silhouette_sample_size: Optional[int] = 10000,
) -> Dict[str, float]:
    """
    Métricas internas estándar para evaluar un clustering.

    Parameters
    ----------
    X : array [n_samples, 2] (x_m, y_m)
    labels : etiquetas de cluster
    silhouette_sample_size : límite superior de muestras para calcular Silhouette

    Returns
    -------
    dict: {'silhouette': float, 'dbi': float, 'chi': float}
    """
    sil = _safe_silhouette(X, labels, random_state=random_state, sample_size=silhouette_sample_size)
    dbi = _safe_davies_bouldin(X, labels)
    chi = _safe_calinski_harabasz(X, labels)
    return {"silhouette": sil, "dbi": dbi, "chi": chi}


def fit_kmeans(
    X: np.ndarray,
    k: int,
    use_minibatch: bool = True,
    random_state: int = 42,
    n_init: int | str = "auto",
    max_iter: int = 300,
    batch_size: int = 2048,
) -> Tuple[object, np.ndarray, np.ndarray]:
    """
    Entrena K-Means (o MiniBatchKMeans) y devuelve modelo, etiquetas y centroides.
    """
    if use_minibatch:
        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=n_init,
            batch_size=batch_size,
            max_iter=max_iter,
            verbose=0,
        )
    else:
        km = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
            verbose=0,
        )
    labels = km.fit_predict(X)
    centroids = km.cluster_centers_
    return km, labels, centroids


def _labels_from_model(model, X: np.ndarray) -> np.ndarray:
    """
    Asigna clusters a X con el modelo entrenado (via predict).
    """
    return model.predict(X)


def bootstrap_kmeans_stability(
    X: np.ndarray,
    k: int,
    n_boot: int = 10,
    sample_frac: float = 0.8,
    random_state: int = 42,
    use_minibatch: bool = True,
    n_init: int | str = "auto",
    max_iter: int = 300,
    batch_size: int = 2048,
) -> Dict[str, float]:
    """
    Estima la estabilidad de K-Means via bootstrap con ARI:
      - Entrena un modelo de referencia en el 100% de X.
      - Para cada bootstrap:
          * toma una submuestra (80% por defecto),
          * entrena un nuevo modelo,
          * compara etiquetas del modelo referencia vs. bootstrap EN LA SUBMUESTRA
            usando Adjusted Rand Index (ARI, invariante a permutaciones de etiquetas).

    Returns
    -------
    dict: {'mean_ari': float, 'std_ari': float}
    """
    rng = check_random_state(random_state)

    # Modelo de referencia (full data)
    ref_model, _, _ = fit_kmeans(
        X, k,
        use_minibatch=use_minibatch,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        batch_size=batch_size,
    )
    ref_labels_full = _labels_from_model(ref_model, X)

    aris = []
    n = X.shape[0]
    m = int(np.clip(int(sample_frac * n), 2 * k, n))  # asegura > 2k puntos para robustez

    for b in range(n_boot):
        idx = rng.choice(n, size=m, replace=False)
        Xb = X[idx]

        boot_model, _, _ = fit_kmeans(
            Xb, k,
            use_minibatch=use_minibatch,
            random_state=rng.randint(0, 10_000_000),
            n_init=n_init,
            max_iter=max_iter,
            batch_size=batch_size,
        )
        # Etiquetas comparables en el mismo subconjunto
        y_ref = ref_labels_full[idx]
        y_boot = _labels_from_model(boot_model, Xb)
        ari = adjusted_rand_score(y_ref, y_boot)
        aris.append(ari)

    aris = np.asarray(aris, dtype=float)
    return {"mean_ari": float(np.nanmean(aris)), "std_ari": float(np.nanstd(aris, ddof=1) if len(aris) > 1 else 0.0)}


def _normalize_series(s: pd.Series, higher_is_better: bool) -> pd.Series:
    """
    Min–max normalization. Si 'higher_is_better' es False, invierte el sentido.
    Devuelve NaN si varianza cero; caller debe manejar NaN.
    """
    s = s.copy()
    if not higher_is_better:
        s = -s
    s_min, s_max = s.min(), s.max()
    if not np.isfinite(s_min) or not np.isfinite(s_max) or np.isclose(s_min, s_max):
        return pd.Series(np.nan, index=s.index)
    return (s - s_min) / (s_max - s_min)


def select_best_k(
    df_metrics: pd.DataFrame,
    weights: Dict[str, float] = None,
    add_column: bool = True
) -> int:
    """
    Selecciona K mediante un score compuesto.
    Por defecto, pondera Silhouette y ARI de manera balanceada.

    weights por defecto:
      - silhouette: 0.40 (↑ mejor)
      - mean_ari  : 0.40 (↑ mejor)
      - chi       : 0.15 (↑ mejor)
      - dbi       : 0.05 (↓ mejor)

    Devuelve:
      - best_k (y añade 'composite_score' si add_column=True)
    """
    if weights is None:
        weights = {"silhouette": 0.40, "mean_ari": 0.40, "chi": 0.15, "dbi": 0.05}

    df = df_metrics.copy()

    # Normalizaciones (min-max)
    nsil = _normalize_series(df["silhouette"], higher_is_better=True)
    nari = _normalize_series(df["mean_ari"], higher_is_better=True)
    nchi = _normalize_series(df["chi"], higher_is_better=True)
    ndbi = _normalize_series(df["dbi"], higher_is_better=False)

    comp = (
        weights.get("silhouette", 0) * nsil.fillna(0) +
        weights.get("mean_ari", 0)  * nari.fillna(0) +
        weights.get("chi", 0)       * nchi.fillna(0) +
        weights.get("dbi", 0)       * ndbi.fillna(0)
    )

    if add_column:
        df["composite_score"] = comp
        # Devuelve el K del máximo score compuesto
        best_k = int(df.loc[df["composite_score"].idxmax(), "k"])
        # Empata con CH como desempate secundario si hay NaNs generalizados
        if not np.isfinite(df["composite_score"].max()):
            best_k = int(df.loc[df["chi"].idxmax(), "k"])
        return best_k
    else:
        # Si no se añade columna, solo devuelve mejor K por comp
        if np.all(~np.isfinite(comp)):
            return int(df.loc[df["chi"].idxmax(), "k"])
        return int(df.loc[comp.idxmax(), "k"])


def search_kmeans_k(
    df: pd.DataFrame,
    x_col: str = "x_m",
    y_col: str = "y_m",
    k_list: List[int] = (8, 12, 16, 20, 24, 28, 32),
    use_minibatch: bool = True,
    random_state: int = 42,
    n_init: int | str = "auto",
    max_iter: int = 300,
    batch_size: int = 2048,
    silhouette_sample_size: Optional[int] = 10000,
    n_boot: int = 10,
    sample_frac: float = 0.8,
    weights: Dict[str, float] = None,
) -> Tuple[pd.DataFrame, int]:
    """
    Pipeline de búsqueda de K:
      - Entrena (MiniBatch)KMeans para cada K en k_list.
      - Calcula métricas internas (Silhouette, DBI, CH).
      - Calcula estabilidad bootstrap (mean_ari, std_ari) por K.
      - Devuelve tabla df_metrics y K recomendado por 'select_best_k'.

    Returns
    -------
    (df_metrics, best_k)
    df_metrics: DataFrame con columnas:
        ['k', 'silhouette', 'dbi', 'chi', 'mean_ari', 'std_ari']
    """
    # Validaciones básicas
    assert x_col in df.columns and y_col in df.columns, f"Faltan columnas {x_col}/{y_col}"
    X = df[[x_col, y_col]].to_numpy(dtype=float)
    # Para reproducibilidad controlada, barajar de forma determinista si se quiere submuestrear internamente
    rng = check_random_state(random_state)

    rows = []
    for k in k_list:
        # Modelo "principal" para métricas internas
        model, labels, _ = fit_kmeans(
            X, k,
            use_minibatch=use_minibatch,
            random_state=random_state,
            n_init=n_init,
            max_iter=max_iter,
            batch_size=batch_size,
        )
        m = compute_internal_metrics(
            X, labels,
            random_state=random_state,
            silhouette_sample_size=silhouette_sample_size
        )
        # Estabilidad bootstrap (comparación con modelo full)
        stab = bootstrap_kmeans_stability(
            X, k,
            n_boot=n_boot,
            sample_frac=sample_frac,
            random_state=random_state,
            use_minibatch=use_minibatch,
            n_init=n_init,
            max_iter=max_iter,
            batch_size=batch_size,
        )

        rows.append({
            "k": int(k),
            "silhouette": m["silhouette"],
            "dbi": m["dbi"],
            "chi": m["chi"],
            "mean_ari": stab["mean_ari"],
            "std_ari": stab["std_ari"],
        })

    df_metrics = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)

    # Selección de K por criterio compuesto
    best_k = select_best_k(df_metrics, weights=weights, add_column=True)
    return df_metrics, best_k







def train_final_kmeans(
    df: pd.DataFrame,
    k: int,
    x_col: str = "x_m",
    y_col: str = "y_m",
    use_minibatch: bool = True,
    random_state: int = 42,
    n_init: int | str = "auto",
    max_iter: int = 300,
    batch_size: int = 2048,
) -> Tuple[object, np.ndarray]:
    """
    Entrena el modelo final de K-Means sobre el dataset completo de entrenamiento.
    Devuelve el modelo y los centroides.
    """
    X = df[[x_col, y_col]].to_numpy(dtype=float)
    model, labels, centroids = fit_kmeans(
        X,
        k,
        use_minibatch=use_minibatch,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        batch_size=batch_size,
    )
    df["cluster_id"] = labels
    return model, centroids


def assign_clusters_to_new_points(
    model,
    df_new: pd.DataFrame,
    x_col: str = "x_m",
    y_col: str = "y_m",
) -> pd.DataFrame:
    """
    Asigna clusters a nuevos puntos según centroides del modelo entrenado.
    """
    X_new = df_new[[x_col, y_col]].to_numpy(dtype=float)
    labels = model.predict(X_new)
    out = df_new.copy()
    out["cluster_id"] = labels
    return out


def evaluate_temporal_consistency(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    cluster_col: str = "cluster_id",
    category_col: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evalúa estabilidad básica de los clusters entre train y test:
      - Cambio en frecuencia relativa de clusters
      - (opcional) KL-divergence entre distribuciones de categorías por cluster
    """
    from scipy.stats import entropy

    # Distribución de clusters
    p_train = df_train[cluster_col].value_counts(normalize=True).sort_index()
    p_test = df_test[cluster_col].value_counts(normalize=True).reindex(p_train.index, fill_value=0)
    kl_clusters = float(entropy(p_train + 1e-9, p_test + 1e-9))

    out = {"kl_clusters": kl_clusters}

    # Distribución de categorías dentro de clusters (opcional)
    if category_col and category_col in df_train.columns and category_col in df_test.columns:
        kls = []
        for cid in sorted(df_train[cluster_col].unique()):
            p1 = (
                df_train.loc[df_train[cluster_col] == cid, category_col]
                .value_counts(normalize=True)
                .sort_index()
            )
            p2 = (
                df_test.loc[df_test[cluster_col] == cid, category_col]
                .value_counts(normalize=True)
                .reindex(p1.index, fill_value=0)
            )
            kls.append(entropy(p1 + 1e-9, p2 + 1e-9))
        out["kl_categories_mean"] = float(np.mean(kls))
        out["kl_categories_std"] = float(np.std(kls, ddof=1))
    return out


def save_geo_kmeans_artifacts(
    model,
    centroids: np.ndarray,
    results_dir: str,
    prefix: str = "geo_kmeans",
    center_lat: float = 19.4326,
    center_lon: float = -99.1332,
    crs_method: str = "aeqd",
    best_k: Optional[int] = None,
) -> Dict[str, str]:
    """
    Guarda:
      - modelo entrenado (.joblib)
      - centroides (.csv)
      - metadatos (.json)
    """
    import json
    os.makedirs(results_dir, exist_ok=True)

    model_path = os.path.join(results_dir, f"{prefix}_model.joblib")
    centroids_path = os.path.join(results_dir, f"{prefix}_centroids.csv")
    meta_path = os.path.join(results_dir, f"{prefix}_meta.json")

    joblib.dump(model, model_path)
    pd.DataFrame(centroids, columns=["x_m", "y_m"]).to_csv(centroids_path, index=False)

    meta = {
        "center_lat": center_lat,
        "center_lon": center_lon,
        "crs_method": crs_method,
        "best_k": best_k,
        "n_centroids": len(centroids),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return {"model": model_path, "centroids": centroids_path, "meta": meta_path}






# --- Inversa de proyección ---
def inverse_project_to_latlon(
    df: pd.DataFrame,
    x_col: str = "x_m",
    y_col: str = "y_m",
    center_lat: float = 19.4326,
    center_lon: float = -99.1332,
    method: str = "aeqd",
    lat_name: str = "lat",
    lon_name: str = "lon",
    inplace: bool = False
) -> pd.DataFrame:
    """
    Convierte coordenadas en metros (CRS local) de vuelta a lat/lon (WGS84).
    Requiere pyproj.
    """
    if not _PYPROJ_AVAILABLE:
        raise ImportError("pyproj no está disponible para inverse_project_to_latlon.")
    assert_required_columns(df, [x_col, y_col], context="inverse_project_to_latlon")

    crs_src = CRS.from_proj4(get_local_crs(center_lat, center_lon, method=method))
    crs_dst = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(crs_src, crs_dst, always_xy=True)

    lons, lats = transformer.transform(df[x_col].values, df[y_col].values)
    if inplace:
        df[lat_name] = lats
        df[lon_name] = lons
        return df
    out = df.copy()
    out[lat_name] = lats
    out[lon_name] = lons
    return out


# --- Selección de K: plots de métricas ---
def plot_k_selection(
    df_metrics: pd.DataFrame,
    save_dir: str,
    filename_prefix: str = "k_selection"
) -> Dict[str, str]:
    """
    Produce 3 figuras:
      - silhouette_vs_k.png
      - dbi_vs_k.png
      - chi_vs_k.png
    Si existe 'composite_score' la grafica también (composite_vs_k.png).
    """
    os.makedirs(save_dir, exist_ok=True)
    paths = {}

    # Silhouette
    fig = plt.figure(figsize=(6,4))
    plt.plot(df_metrics["k"], df_metrics["silhouette"], marker="o")
    plt.xlabel("K"); plt.ylabel("Silhouette ↑"); plt.title("Silhouette vs K")
    out1 = os.path.join(save_dir, f"{filename_prefix}_silhouette_vs_k.png")
    fig.tight_layout(); fig.savefig(out1, dpi=140); plt.close(fig)
    paths["silhouette"] = out1

    # DBI
    fig = plt.figure(figsize=(6,4))
    plt.plot(df_metrics["k"], df_metrics["dbi"], marker="o")
    plt.xlabel("K"); plt.ylabel("Davies–Bouldin ↓"); plt.title("DBI vs K")
    out2 = os.path.join(save_dir, f"{filename_prefix}_dbi_vs_k.png")
    fig.tight_layout(); fig.savefig(out2, dpi=140); plt.close(fig)
    paths["dbi"] = out2

    # CH
    fig = plt.figure(figsize=(6,4))
    plt.plot(df_metrics["k"], df_metrics["chi"], marker="o")
    plt.xlabel("K"); plt.ylabel("Calinski–Harabasz ↑"); plt.title("CH vs K")
    out3 = os.path.join(save_dir, f"{filename_prefix}_chi_vs_k.png")
    fig.tight_layout(); fig.savefig(out3, dpi=140); plt.close(fig)
    paths["chi"] = out3

    # Composite (si existe)
    if "composite_score" in df_metrics.columns:
        fig = plt.figure(figsize=(6,4))
        plt.plot(df_metrics["k"], df_metrics["composite_score"], marker="o")
        plt.xlabel("K"); plt.ylabel("Score compuesto ↑"); plt.title("Score compuesto vs K")
        out4 = os.path.join(save_dir, f"{filename_prefix}_composite_vs_k.png")
        fig.tight_layout(); fig.savefig(out4, dpi=140); plt.close(fig)
        paths["composite"] = out4

    return paths


# --- Mapas de clusters ---
def make_cluster_map(
    df_points: pd.DataFrame,
    centroids_xy: np.ndarray,
    save_dir: str,
    center_lat: float = 19.4326,
    center_lon: float = -99.1332,
    crs_method: str = "aeqd",
    lat_col_out: str = "lat",
    lon_col_out: str = "lon",
    cluster_col: str = "cluster_id",
    filename_html: str = "clusters_map.html"
) -> str:
    """
    Genera un mapa interactivo HTML con Folium. Si Folium no está disponible, guarda un scatter estático PNG.
    - df_points: debe tener x_m, y_m y cluster_id
    - centroids_xy: array [K,2] en metros
    """
    os.makedirs(save_dir, exist_ok=True)
    # Convertir puntos y centroides a lat/lon
    pts_latlon = inverse_project_to_latlon(
        df_points, x_col="x_m", y_col="y_m",
        center_lat=center_lat, center_lon=center_lon, method=crs_method,
        lat_name=lat_col_out, lon_name=lon_col_out, inplace=False
    )
    cent_df = pd.DataFrame(centroids_xy, columns=["x_m","y_m"])
    cent_df = inverse_project_to_latlon(
        cent_df, x_col="x_m", y_col="y_m",
        center_lat=center_lat, center_lon=center_lon, method=crs_method,
        lat_name=lat_col_out, lon_name=lon_col_out, inplace=False
    )
    try:
        import folium
        from folium.plugins import MarkerCluster
        fmap = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB positron")
        # Marcadores agrupados para puntos
        mc = MarkerCluster()
        for _, r in pts_latlon.sample(min(len(pts_latlon), 5000), random_state=42).iterrows():
            folium.CircleMarker(
                location=[r[lat_col_out], r[lon_col_out]],
                radius=2,
                popup=str(r.get(cluster_col, "")),
                opacity=0.6, fill=True, fill_opacity=0.6
            ).add_to(mc)
        mc.add_to(fmap)
        # Centroides
        for i, r in cent_df.iterrows():
            folium.Marker(
                location=[r[lat_col_out], r[lon_col_out]],
                tooltip=f"centroid {i}",
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(fmap)
        out_html = os.path.join(save_dir, filename_html)
        fmap.save(out_html)
        return out_html
    except Exception:
        # Fallback estático
        fig = plt.figure(figsize=(6,6))
        plt.scatter(pts_latlon[lon_col_out], pts_latlon[lat_col_out], s=3, alpha=0.3)
        plt.scatter(cent_df[lon_col_out], cent_df[lat_col_out], s=80, marker="x")
        plt.xlabel("lon"); plt.ylabel("lat"); plt.title("Clusters (vista estática)")
        out_png = os.path.join(save_dir, filename_html.replace(".html",".png"))
        fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)
        return out_png


# --- Perfiles de cluster: distancia a centroide y composición ---
def _compute_distance_to_centroid(
    df: pd.DataFrame,
    centroids_xy: np.ndarray,
    x_col: str = "x_m",
    y_col: str = "y_m",
    cluster_col: str = "cluster_id"
) -> pd.Series:
    """
    Distancia euclídea en metros de cada punto a su centroide asignado.
    """
    assert_required_columns(df, [x_col, y_col, cluster_col], context="_compute_distance_to_centroid")
    arr = df[[x_col, y_col]].to_numpy(float)
    cids = df[cluster_col].to_numpy(int)
    dists = np.linalg.norm(arr - centroids_xy[cids], axis=1)
    return pd.Series(dists, index=df.index, name="dist_to_centroid_m")


def plot_cluster_profiles(
    df: pd.DataFrame,
    centroids_xy: np.ndarray,
    save_dir: str,
    cluster_col: str = "cluster_id",
    category_col: Optional[str] = None,
    filename_prefix: str = "cluster_profiles"
) -> Dict[str, str]:
    """
    Genera:
      - Boxplot de distancias a centroide por cluster
      - (Opcional) Barras apiladas de composición categórica por cluster
    """
    os.makedirs(save_dir, exist_ok=True)
    # Distancias
    dists = _compute_distance_to_centroid(df, centroids_xy, cluster_col=cluster_col)
    tmp = pd.concat([df[[cluster_col]], dists], axis=1)

    fig = plt.figure(figsize=(10,5))
    tmp.boxplot(column="dist_to_centroid_m", by=cluster_col, grid=False, showfliers=False)
    plt.suptitle(""); plt.title("Distancia a centroide por cluster (m)")
    plt.xlabel("cluster_id"); plt.ylabel("distancia (m)")
    out1 = os.path.join(save_dir, f"{filename_prefix}_distance_boxplot.png")
    fig.tight_layout(); fig.savefig(out1, dpi=150); plt.close(fig)

    paths = {"distance_boxplot": out1}

    # Composición categórica (si aplica)
    if category_col and category_col in df.columns:
        comp = (
            df.groupby(cluster_col)[category_col]
              .value_counts(normalize=True)
              .rename("prop")
              .reset_index()
        )
        # Pivot para barras apiladas
        pv = comp.pivot(index=cluster_col, columns=category_col, values="prop").fillna(0.0)
        ax = pv.plot(kind="bar", stacked=True, figsize=(10,5), legend=True)
        ax.set_xlabel("cluster_id"); ax.set_ylabel("proporción"); ax.set_title("Composición por categoría")
        fig = ax.get_figure()
        out2 = os.path.join(save_dir, f"{filename_prefix}_category_composition.png")
        fig.tight_layout(); fig.savefig(out2, dpi=150); plt.close(fig)
        paths["category_composition"] = out2

    return paths


# --- Series temporales por cluster ---
def aggregate_timeseries_by_cluster(
    df: pd.DataFrame,
    date_col: str,
    cluster_col: str = "cluster_id",
    freq: str = "W"
) -> pd.DataFrame:
    """
    Agrega conteos por cluster y periodo (freq='W' semanal por defecto).
    Retorna un DataFrame pivotado: index=periodo, columnas=cluster_id, valores=conteo.
    """
    assert_required_columns(df, [date_col, cluster_col], context="aggregate_timeseries_by_cluster")
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        raise TypeError(f"{date_col} debe ser datetime64[ns]")
    g = (
        df.set_index(date_col)
          .groupby(cluster_col)
          .resample(freq)
          .size()
          .rename("count")
          .reset_index()
    )
    pv = g.pivot(index=date_col, columns=cluster_col, values="count").fillna(0).sort_index()
    return pv


def plot_cluster_timeseries(
    ts_df: pd.DataFrame,
    save_dir: str,
    top_n: int = 8,
    filename_prefix: str = "cluster_timeseries"
) -> str:
    """
    Dibuja las series temporales de los clusters (hasta top_n por promedio).
    """
    os.makedirs(save_dir, exist_ok=True)
    means = ts_df.mean(axis=0).sort_values(ascending=False)
    top_cols = means.head(min(top_n, ts_df.shape[1])).index.tolist()
    fig = plt.figure(figsize=(10,5))
    ts_df[top_cols].plot(ax=plt.gca())
    plt.title(f"Series temporales semanales por cluster (top {len(top_cols)})")
    plt.xlabel("fecha"); plt.ylabel("conteo")
    out = os.path.join(save_dir, f"{filename_prefix}_weekly.png")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out


# --- Reporte Markdown rápido ---
def write_geo_report_markdown(
    save_dir: str,
    best_k: int,
    paths_k_selection: Dict[str, str],
    map_path: str,
    profile_paths: Dict[str, str],
    ts_path: str,
    consistency: Optional[Dict[str, float]] = None,
    extra_notes: Optional[str] = None,
    filename: str = "report_geo_clustering.md"
) -> str:
    """
    Genera un reporte Markdown con enlaces a las figuras/artefactos.
    """
    os.makedirs(save_dir, exist_ok=True)
    lines = []
    lines.append(f"# Geo-Clustering Report\n")
    lines.append(f"- **BEST_K:** {best_k}\n")
    if consistency:
        for k, v in consistency.items():
            lines.append(f"- **{k}**: {v:.4f}")
        lines.append("")

    lines.append("## Selección de K (métricas)")
    for k, p in paths_k_selection.items():
        lines.append(f"- {k}: `{p}`")

    lines.append("\n## Mapa de clusters")
    lines.append(f"- mapa: `{map_path}`")

    lines.append("\n## Perfiles de clusters")
    for k, p in profile_paths.items():
        lines.append(f"- {k}: `{p}`")

    lines.append("\n## Series temporales")
    lines.append(f"- semanales: `{ts_path}`")

    if extra_notes:
        lines.append("\n## Notas")
        lines.append(extra_notes)

    out_md = os.path.join(save_dir, filename)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_md





def make_cluster_map_inline(
    df_points: pd.DataFrame,
    centroids_xy: np.ndarray,
    center_lat: float = 19.4326,
    center_lon: float = -99.1332,
    crs_method: str = "aeqd",
    cluster_col: str = "cluster_id",
    lat_col_out: str = "lat",
    lon_col_out: str = "lon",
    max_points: int = 5000,
    random_state: int = 42
):
    """
    Genera un mapa de clusters que se renderiza directamente en el notebook.
    - Si 'folium' está disponible, retorna un objeto folium.Map (se renderiza inline).
    - Si 'folium' no está disponible, retorna una figura de Matplotlib.

    Parámetros
    ----------
    df_points : DataFrame con columnas ['x_m','y_m', cluster_col]
    centroids_xy : np.ndarray de forma [K,2] en metros (x_m, y_m)
    center_lat, center_lon : centro del CRS local y vista del mapa
    crs_method : 'aeqd' | 'laea' | 'lcc'
    cluster_col : nombre de la columna de cluster
    lat_col_out, lon_col_out : nombres de columnas lat/lon generadas para visualización
    max_points : cantidad máxima de puntos a muestrear para el mapa (mejora rendimiento)
    random_state : semilla para muestreo reproducible

    Retorna
    -------
    - folium.Map si folium está disponible
    - matplotlib.figure.Figure si no lo está
    """
    assert_required_columns(df_points, ["x_m", "y_m", cluster_col], context="make_cluster_map_inline")

    # Convertir puntos y centroides a lat/lon
    pts_latlon = inverse_project_to_latlon(
        df_points, x_col="x_m", y_col="y_m",
        center_lat=center_lat, center_lon=center_lon, method=crs_method,
        lat_name=lat_col_out, lon_name=lon_col_out, inplace=False
    )
    cent_df = pd.DataFrame(centroids_xy, columns=["x_m","y_m"])
    cent_df = inverse_project_to_latlon(
        cent_df, x_col="x_m", y_col="y_m",
        center_lat=center_lat, center_lon=center_lon, method=crs_method,
        lat_name=lat_col_out, lon_name=lon_col_out, inplace=False
    )

    # Muestreo para rendimiento
    if len(pts_latlon) > max_points:
        pts_latlon = pts_latlon.sample(max_points, random_state=random_state)

    # Intentar folium
    try:
        import folium
        from folium.plugins import MarkerCluster

        fmap = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB positron")

        # Capa de puntos (usamos círculos por rendimiento)
        mc = MarkerCluster(name="Eventos")
        for _, r in pts_latlon.iterrows():
            folium.CircleMarker(
                location=[r[lat_col_out], r[lon_col_out]],
                radius=2,
                opacity=0.6,
                fill=True,
                fill_opacity=0.6,
                popup=f"{cluster_col}: {r.get(cluster_col, '')}"
            ).add_to(mc)
        mc.add_to(fmap)

        # Capa de centroides
        cent_layer = folium.FeatureGroup(name="Centroides", show=True)
        for i, r in cent_df.iterrows():
            folium.Marker(
                location=[r[lat_col_out], r[lon_col_out]],
                tooltip=f"centroid {i}",
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(cent_layer)
        cent_layer.add_to(fmap)

        folium.LayerControl().add_to(fmap)
        return fmap

    except Exception:
        # Fallback: figura estática
        fig = plt.figure(figsize=(6, 6))
        plt.scatter(pts_latlon[lon_col_out], pts_latlon[lat_col_out], s=3, alpha=0.3, label="puntos")
        plt.scatter(cent_df[lon_col_out], cent_df[lat_col_out], s=80, marker="x", label="centroides")
        plt.xlabel("lon"); plt.ylabel("lat")
        plt.title("Clusters (vista estática)")
        plt.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        return fig






def plot_cluster_timeseries_inline(
    df: pd.DataFrame,
    date_col: str,
    cluster_col: str = "cluster_id",
    freq: str = "W",
    start_year: int = 2010,
    top_n: int = 8
):
    """
    Agrega y visualiza las series temporales por cluster directamente en notebook.

    - Filtra los datos desde el año 'start_year' en adelante.
    - Muestra el gráfico inline (no guarda archivo).
    """
    assert_required_columns(df, [date_col, cluster_col], context="plot_cluster_timeseries_inline")

    # Asegurar tipo datetime
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        raise TypeError(f"{date_col} debe ser datetime64[ns]")

    # Filtro temporal
    df_f = df[df[date_col] >= pd.Timestamp(f"{start_year}-01-01")].copy()
    if df_f.empty:
        raise ValueError(f"No hay datos desde el año {start_year} en adelante.")

    # Agregación semanal
    ts_df = (
        df_f.set_index(date_col)
           .groupby(cluster_col)
           .resample(freq)
           .size()
           .rename("count")
           .reset_index()
           .pivot(index=date_col, columns=cluster_col, values="count")
           .fillna(0)
           .sort_index()
    )

    # Selección de clusters más activos
    means = ts_df.mean(axis=0).sort_values(ascending=False)
    top_cols = means.head(min(top_n, ts_df.shape[1])).index.tolist()

    # Plot inline
    fig, ax = plt.subplots(figsize=(10, 5))
    ts_df[top_cols].plot(ax=ax)
    ax.set_title(f"Series temporales semanales por cluster (top {len(top_cols)}) — desde {start_year}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Conteo de eventos")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return ts_df  # devuelve el DataFrame agregado para análisis posterior













##########################################
### PREDICCIÓN EN SERIES DE TIEMPO
##########################################

# helpers para conflictos de versión de h3 v3/v4



def _h3_latlon_to_cell(lat: float, lon: float, res: int) -> str:
    """
    Wrapper compatible con h3 v3/v4.
    v3: h3.geo_to_h3(lat, lng, resolution)
    v4: h3.latlng_to_cell(lat, lng, res)
    """
    if h3 is None:
        raise RuntimeError("h3 no disponible.")
    # v4
    if hasattr(h3, "latlng_to_cell"):
        return h3.latlng_to_cell(lat, lon, res)
    # v3
    if hasattr(h3, "geo_to_h3"):
        return h3.geo_to_h3(lat=lat, lng=lon, resolution=res)
    raise RuntimeError("No se encontró una función válida para convertir lat/lon a celda en h3.")


def _h3_cell_to_latlon(cell_id: str) -> tuple[float, float]:
    """
    Devuelve (lon, lat) del centroide de la celda, compatible v3/v4.
    v3: h3.h3_to_geo(cell) -> (lat, lon)
    v4: h3.cell_to_latlng(cell) -> (lat, lon)
    """
    if h3 is None:
        raise RuntimeError("h3 no disponible.")
    # v4
    if hasattr(h3, "cell_to_latlng"):
        lat, lon = h3.cell_to_latlng(cell_id)
        return float(lon), float(lat)
    # v3
    if hasattr(h3, "h3_to_geo"):
        lat, lon = h3.h3_to_geo(cell_id)
        return float(lon), float(lat)
    raise RuntimeError("No se encontró una función válida para obtener centroide de celda en h3.")






def _check_cols(df: pd.DataFrame, cols: Iterable[str]) -> None:
    """Verifica columnas obligatorias en df."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en df: {missing}")



def haversine_distance_m(lat1, lon1, lat2, lon2) -> float:
    """
    Distancia haversine en metros entre dos puntos (lat/lon en grados).
    Útil para métricas espaciales (Hit@k dentro de radio R).
    """
    R = 6371000.0  # radio medio de la tierra [m]
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return float(R * c)



# Configuración de ventana N (2–6 días) de predicción
def check_window_N(N: int) -> None:
    """
    Valida el horizonte N (en días): 2 ≤ N < 7.
    Esta verificación centraliza la regla de negocio.
    """
    if not isinstance(N, int):
        raise TypeError("N debe ser entero (días).")
    if N < 2 or N >= 7:
        raise ValueError("N debe cumplir: 2 ≤ N < 7.")







# Discretización espacial: asignación de celdas (H3 / geohash)
def index_points_to_cells(
    df: pd.DataFrame,
    lon_col: str = "lon",
    lat_col: str = "lat",
    method: str = "h3",
    h3_res: int = 8,
    geohash_prec: int = 6,
    out_col: str = "cell_id",
) -> pd.DataFrame:
    """
    Discretiza coordenadas (lon/lat) en celdas espaciales.
    - Mantiene el mismo número de filas del df de entrada.
    - Asigna NaN en `cell_id` cuando la coordenada es inválida (NA, no-numérica o fuera de rango).
    """
    _check_cols(df, [lon_col, lat_col])
    out = df.copy()

    # Convertir a numérico con coerce (soporta Float64 con NA)
    lon_vals = pd.to_numeric(out[lon_col], errors="coerce")
    lat_vals = pd.to_numeric(out[lat_col], errors="coerce")

    # Máscara de coordenadas válidas
    valid = (
        lon_vals.notna() & lat_vals.notna()
        & np.isfinite(lon_vals) & np.isfinite(lat_vals)
        & lon_vals.between(-180.0, 180.0)
        & lat_vals.between(-90.0, 90.0)
    )

    # Inicializar columna de salida con NaN (object para strings de cell_id)
    out[out_col] = pd.Series([np.nan] * len(out), dtype="object")

    if method == "h3":
        if h3 is None:
            raise RuntimeError("No se encontró la librería 'h3'. Instala con: pip install h3")
        # Solo calcular para filas válidas
        res_vals = [
            _h3_latlon_to_cell(lat=float(la), lon=float(lo), res=int(h3_res))
            for lo, la in zip(lon_vals[valid].values, lat_vals[valid].values)
        ]
        out.loc[valid, out_col] = res_vals

    elif method == "geohash":
        if pgh is None:
            raise RuntimeError("No se encontró 'pygeohash'. Instala con: pip install pygeohash")
        res_vals = [
            pgh.encode(float(la), float(lo), precision=int(geohash_prec))
            for lo, la in zip(lon_vals[valid].values, lat_vals[valid].values)
        ]
        out.loc[valid, out_col] = res_vals

    else:
        raise ValueError("method debe ser 'h3' o 'geohash'.")

    # logging simple de cuántas filas quedaron sin cell_id
    n_invalid = int((~valid).sum())
    if n_invalid > 0:
        print(f"[INFO] index_points_to_cells: {n_invalid} filas con coordenadas inválidas → cell_id=NaN")

    return out





def cell_centroid(cell_id: str, method: str = "h3") -> tuple[float, float]:
    """
    Devuelve el centroide aproximado de la celda (lon, lat).
    """
    if method == "h3":
        return _h3_cell_to_latlon(cell_id)
    elif method == "geohash":
        if pgh is None:
            raise RuntimeError("pygeohash no disponible.")
        lat, lon = pgh.decode(cell_id)
        return float(lon), float(lat)
    else:
        raise ValueError("method debe ser 'h3' o 'geohash'.")






def cells_to_centroids(
    cells: Iterable[str],
    method: str = "h3",
) -> pd.DataFrame:
    """
    Convierte una lista de cell_id a un DataFrame con centroides (lon, lat).
    """
    data = []
    for cid in cells:
        lon, lat = cell_centroid(cid, method=method)
        data.append({"cell_id": cid, "lon": lon, "lat": lat})
    return pd.DataFrame(data)






# Celdas activas: evitar solamente ceros con umbral de actividad X
def build_active_cells(
    df_events: pd.DataFrame,
    cell_col: str = "cell_id",
    cat_col: Optional[str] = "categoria_delito",
    min_events_total: int = 10,
    min_events_per_cat: int = 3,
    per_category: bool = True,
) -> pd.DataFrame:
    """
    Construye el catálogo de celdas activas para modelado:
      - per_category=True: asegura que cada (cell, cat) tenga al menos `min_events_per_cat`.
      - per_category=False: activa una celda si su total ≥ min_events_total (independiente de categoría).

    Devuelve:
      - DataFrame con columnas:
          cell_id, categoria_delito (si per_category), total_events
    """
    _check_cols(df_events, [cell_col])
    df = df_events.copy()

    if per_category:
        if cat_col is None or cat_col not in df.columns:
            raise ValueError("cat_col es requerido cuando per_category=True.")
        g = (df.groupby([cell_col, cat_col])
                .size()
                .rename("total_events")
                .reset_index())
        act = g[g["total_events"] >= min_events_per_cat].reset_index(drop=True)
        return act
    else:
        g = df.groupby(cell_col).size().rename("total_events").reset_index()
        act = g[g["total_events"] >= min_events_total].reset_index(drop=True)
        return act




# Top-k hotspots: de scores por (cell, cat) a lat/lon
def rank_hotspots_topk(
    scores_df: pd.DataFrame,
    k: int = 20,
    score_col: str = "score",
    cell_col: str = "cell_id",
    cat_col: str = "categoria_delito",
    method: str = "h3",
) -> pd.DataFrame:
    """
    A partir de un DataFrame con puntajes por (cell, categoría), devuelve el TOP-k
    por categoría, agregando los centroides (lon, lat).

    Espera columnas:
      - cell_id, categoria_delito, score
    """
    _check_cols(scores_df, [cell_col, cat_col, score_col])
    # TOP-k por categoría
    top_list = []
    for cat, g in scores_df.groupby(cat_col):
        g2 = g.sort_values(score_col, ascending=False).head(k).copy()
        cent = cells_to_centroids(g2[cell_col].tolist(), method=method)
        g2 = g2.merge(cent, on=cell_col, how="left")
        top_list.append(g2)
    return pd.concat(top_list, ignore_index=True)





# Métricas espaciales base: Hit@k dentro de radio R
def hit_at_k_radius(
    topk_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    cat_col: str = "categoria_delito",
    # nombres de columnas en predicciones (top-k)
    pred_lat_col: str = "lat",
    pred_lon_col: str = "lon",
    # nombres de columnas en la verdad (incidentes reales)
    truth_lat_col: str = "latitud",
    truth_lon_col: str = "longitud",
    radius_m: float = 500.0,
) -> pd.DataFrame:
    """
    Calcula Hit@k dentro de un radio R (metros) por categoría.

    Parámetros
    ----------
    topk_df : DataFrame
        Salida de rank_hotspots_topk. Debe contener:
        [cat_col, pred_lat_col, pred_lon_col].
    truth_df : DataFrame
        Eventos reales en la ventana evaluada. Debe contener:
        [cat_col, truth_lat_col, truth_lon_col].
    cat_col : str
        Nombre de la columna de categoría.
    pred_lat_col, pred_lon_col : str
        Nombres de columnas lat/lon en el df de predicciones.
    truth_lat_col, truth_lon_col : str
        Nombres de columnas lat/lon en el df de verdad.
    radius_m : float
        Radio de acierto en metros.

    Retorna
    -------
    DataFrame con columnas:
        [cat_col, n_events, n_hits, hit_rate]
    """
    # Validaciones mínimas
    _check_cols(topk_df, [cat_col, pred_lat_col, pred_lon_col])
    _check_cols(truth_df, [cat_col, truth_lat_col, truth_lon_col])

    # Coerción a numérico y filtrado de filas inválidas
    preds = topk_df.copy()
    preds[pred_lat_col] = pd.to_numeric(preds[pred_lat_col], errors="coerce")
    preds[pred_lon_col] = pd.to_numeric(preds[pred_lon_col], errors="coerce")
    preds = preds[preds[[pred_lat_col, pred_lon_col]].notna().all(axis=1)]

    truth = truth_df.copy()
    truth[truth_lat_col] = pd.to_numeric(truth[truth_lat_col], errors="coerce")
    truth[truth_lon_col] = pd.to_numeric(truth[truth_lon_col], errors="coerce")
    truth = truth[truth[[truth_lat_col, truth_lon_col]].notna().all(axis=1)]

    out_rows = []
    for cat, real in truth.groupby(cat_col):
        pred = preds[preds[cat_col] == cat]
        if pred.empty:
            out_rows.append({cat_col: cat, "n_events": len(real), "n_hits": 0, "hit_rate": 0.0})
            continue

        hits = 0
        # Para cada evento real, verifica si hay algún hotspot a <= R metros
        for _, r in real.iterrows():
            lat_r, lon_r = float(r[truth_lat_col]), float(r[truth_lon_col])
            # corto-circuito si encontramos un hotspot dentro del radio
            in_radius = False
            for _, p in pred.iterrows():
                lat_p, lon_p = float(p[pred_lat_col]), float(p[pred_lon_col])
                d = haversine_distance_m(lat_r, lon_r, lat_p, lon_p)
                if d <= radius_m:
                    in_radius = True
                    break
            hits += int(in_radius)

        n_events = int(len(real))
        hit_rate = float(hits / n_events) if n_events > 0 else 0.0
        out_rows.append({cat_col: cat, "n_events": n_events, "n_hits": int(hits), "hit_rate": hit_rate})

    return pd.DataFrame(out_rows)








# Placeholders de criterios de aceptación (latencia e interfaces)
def check_latency_budget(num_cells: int, num_categories: int, budget_ms: float = 2000.0) -> Dict[str, float]:
    """
    Placeholder simple para estimar si estamos dentro de un presupuesto de latencia.
    (Se refinará en etapas de inferencia real.)
    """
    # Suposición: coste aproximado O(num_cells * num_categories) por paso principal.
    # Ajusta este modelo según tu hardware y tamaño real de features/modelo.
    est_ms = 0.002 * num_cells * num_categories  # 2 microsegundos por par (placeholder)
    return {"estimated_ms": est_ms, "within_budget": bool(est_ms <= budget_ms)}








def _h3_cell_boundary_lonlat(cell_id: str) -> list[tuple[float, float]]:
    """
    Devuelve el contorno de la celda H3 como lista de (lon, lat),
    compatible con h3 v3/v4. El último punto cierra el polígono.
    """
    if h3 is None:
        raise RuntimeError("h3 no disponible.")

    # h3 v4: cell_to_boundary(cell) -> [(lat, lon), ...]
    if hasattr(h3, "cell_to_boundary"):
        boundary = h3.cell_to_boundary(cell_id)  # v4 no admite geo_json kwarg
        pts = [(float(lon), float(lat)) for lat, lon in boundary]
        if pts and pts[0] != pts[-1]:
            pts.append(pts[0])
        return pts

    # h3 v3: h3_to_geo_boundary(cell, geo_json=False) -> [(lat, lon), ...]
    if hasattr(h3, "h3_to_geo_boundary"):
        boundary = h3.h3_to_geo_boundary(cell_id, geo_json=False)
        pts = [(float(lon), float(lat)) for lat, lon in boundary]
        if pts and pts[0] != pts[-1]:
            pts.append(pts[0])
        return pts

    raise RuntimeError("No se halló API de boundary compatible (h3 v3/v4).")







def active_cells_centroids(
    active_cells_df: pd.DataFrame,
    cell_col: str = "cell_id",
    cat_col: str = "categoria_delito",
    events_col: str = "total_events",
    method: str = "h3",
    aggregate_across_categories: bool = True,
) -> pd.DataFrame:
    """
    Construye un DF de centroides por celda.
    Si aggregate_across_categories=True, agrega total_events por celda (suma),
    y cuenta cuántas categorías activas hay en la celda.
    """
    _check_cols(active_cells_df, [cell_col, events_col])
    df = active_cells_df.copy()

    if aggregate_across_categories and cat_col in df.columns:
        agg = (df.groupby(cell_col)
                 .agg({events_col: "sum", cat_col: "nunique"})
                 .rename(columns={events_col: "total_events_sum", cat_col: "n_categories"})
                 .reset_index())
    else:
        agg = (df.groupby(cell_col)
                 .agg({events_col: "sum"})
                 .rename(columns={events_col: "total_events_sum"})
                 .reset_index())
        agg["n_categories"] = np.nan

    cent = cells_to_centroids(agg[cell_col].tolist(), method=method)
    out = agg.merge(cent, on=cell_col, how="left")  # lon, lat
    return out  # columnas: cell_id, total_events_sum, n_categories, lon, lat

def plot_active_cells_static(
    active_cells_df: pd.DataFrame,
    cell_col: str = "cell_id",
    cat_col: str = "categoria_delito",
    events_col: str = "total_events",
    method: str = "h3",
    aggregate_across_categories: bool = True,
    figsize=(8, 8),
) -> "matplotlib.figure.Figure":
    """
    Mapa estático: dispersión de centroides con tamaño ~ total_events_sum.
    Requiere matplotlib importado en el entorno.
    """
    import matplotlib.pyplot as plt

    cent = active_cells_centroids(
        active_cells_df, cell_col=cell_col, cat_col=cat_col, events_col=events_col,
        method=method, aggregate_across_categories=aggregate_across_categories
    )
    if cent.empty:
        raise ValueError("No hay celdas activas para graficar.")

    # Escala de tamaño agradable
    s = cent["total_events_sum"].clip(lower=1)
    s = 20 * (np.log1p(s))  # log para comprimir outliers

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(cent["lon"], cent["lat"], s=s, alpha=0.6)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title("Celdas activas (centroides) — tamaño ~ total de eventos")
    # Etiqueta simple con rango
    tmin, tmed, tmax = cent["total_events_sum"].min(), cent["total_events_sum"].median(), cent["total_events_sum"].max()
    ax.text(0.02, 0.98, f"Eventos (min/med/max): {int(tmin)}/{int(tmed)}/{int(tmax)}",
            transform=ax.transAxes, va="top")
    return fig

def make_folium_map_active_cells(
    active_cells_df: pd.DataFrame,
    cell_col: str = "cell_id",
    cat_col: str = "categoria_delito",
    events_col: str = "total_events",
    method: str = "h3",
    aggregate_across_categories: bool = True,
    max_cells: int = 4000,
    show_polygons: bool = True,
    color_func: Optional[callable] = None,
    tiles: str = "CartoDB positron",
    zoom_start: int = 11,
) -> "folium.Map":
    """
    Mapa interactivo en Folium con celdas activas.
    - Si show_polygons=True y method='h3', dibuja hexágonos H3.
    - Si show_polygons=False, dibuja marcadores en centroides.
    - Trunca a max_cells para evitar mapas pesados.

    color_func: callable(total_events_sum) -> color_hex, p.ej. lambda x: "#ff6600"
                Si None, usa una rampa simple basada en percentiles.
    """
    if folium is None:
        raise RuntimeError("folium no disponible. Instala con: pip install folium")

    cent = active_cells_centroids(
        active_cells_df, cell_col=cell_col, cat_col=cat_col, events_col=events_col,
        method=method, aggregate_across_categories=aggregate_across_categories
    ).sort_values("total_events_sum", ascending=False)

    if cent.empty:
        raise ValueError("No hay celdas activas para mapear.")

    # Centro del mapa: mediana de coords
    ctr_lat = float(cent["lat"].median())
    ctr_lon = float(cent["lon"].median())
    m = folium.Map(location=[ctr_lat, ctr_lon], tiles=tiles, zoom_start=zoom_start)

    # Limitamos para performance
    cent = cent.head(max_cells).reset_index(drop=True)

    # Color por percentiles si no hay color_func
    if color_func is None:
        qs = np.quantile(cent["total_events_sum"], [0.2, 0.5, 0.8])
        def color_func(v):
            if v <= qs[0]: return "#b3cde3"
            if v <= qs[1]: return "#6497b1"
            if v <= qs[2]: return "#005b96"
            return "#03396c"

    for _, r in cent.iterrows():
        v = float(r["total_events_sum"])
        color = color_func(v)

        if show_polygons and method == "h3":
            boundary = _h3_cell_boundary_lonlat(r[cell_col])  # [(lon,lat),...]
            # folium espera (lat, lon)
            poly_latlon = [(lat, lon) for (lon, lat) in boundary]
            folium.Polygon(
                locations=poly_latlon,
                color=color, weight=1, fill=True, fill_color=color, fill_opacity=0.5,
                tooltip=f"{r[cell_col]} | eventos={int(v)} | cats={r['n_categories']}"
            ).add_to(m)
        else:
            folium.CircleMarker(
                location=[r["lat"], r["lon"]],
                radius=6,
                color=color,
                fill=True,
                fill_opacity=0.8,
                tooltip=f"{r[cell_col]} | eventos={int(v)} | cats={r['n_categories']}"
            ).add_to(m)

    return m




# Complementos para mapas: breakdown por categoría + choropleth
def compute_cell_category_breakdown(
    active_cells_df: pd.DataFrame,
    cell_col: str = "cell_id",
    cat_col: str = "categoria_delito",
    events_col: str = "total_events",
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Calcula, para cada celda, el ranking de categorías por total de eventos.
    Devuelve un DataFrame con una fila por celda y columnas:
        [cell_id, cat_breakdown, total_events_sum, n_categories]
    donde:
        - cat_breakdown es una lista de tuplas [(cat, count), ...] ordenada desc y truncada a top_n
        - total_events_sum es la suma total de eventos de la celda (todas las categorías)
        - n_categories es el número de categorías activas de la celda
    """
    _check_cols(active_cells_df, [cell_col, cat_col, events_col])
    df = active_cells_df.copy()

    # agregados por (celda, categoría)
    g = (df.groupby([cell_col, cat_col])[events_col]
            .sum()
            .reset_index()
            .rename(columns={events_col: "events"}))

    # total por celda y #categorías
    totals = (g.groupby(cell_col)["events"].sum().rename("total_events_sum").reset_index())
    ncat = (g.groupby(cell_col)[cat_col].nunique().rename("n_categories").reset_index())
    base = totals.merge(ncat, on=cell_col, how="left")

    # construir top-N por celda
    breakdown_rows = []
    for cid, gg in g.groupby(cell_col):
        gg2 = gg.sort_values("events", ascending=False).head(top_n)
        breakdown = list(zip(gg2[cat_col].tolist(), gg2["events"].astype(int).tolist()))
        row = {
            cell_col: cid,
            "cat_breakdown": breakdown
        }
        breakdown_rows.append(row)

    breakdown_df = pd.DataFrame(breakdown_rows)
    out = base.merge(breakdown_df, on=cell_col, how="left")
    return out  # columnas: [cell_id, total_events_sum, n_categories, cat_breakdown]


def _format_breakdown_html(breakdown: list[tuple[str, int]], max_lines: int = 5) -> str:
    """
    Convierte la lista [(cat, events), ...] a HTML de lista corta para popup.
    """
    if not isinstance(breakdown, list) or len(breakdown) == 0:
        return "<i>Sin detalle</i>"
    lines = []
    for i, (cat, cnt) in enumerate(breakdown[:max_lines], start=1):
        lines.append(f"{i}. {cat}: <b>{cnt}</b>")
    return "<br/>".join(lines)


def _build_quantile_color_fn(series: pd.Series, palette: list[str] | None = None):
    """
    Crea una función color_func(value) que asigna color por cuantiles.
    Devuelve (color_func, breaks) donde breaks son los puntos de corte.
    """
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    # por defecto, 5 cuantiles
    qs = np.quantile(s, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # paleta por defecto (6 niveles)
    if palette is None:
        palette = ["#f1eef6", "#bdc9e1", "#74a9cf", "#2b8cbe", "#045a8d", "#023858"]

    def color_func(v: float) -> str:
        if v <= qs[1]: return palette[0]
        if v <= qs[2]: return palette[1]
        if v <= qs[3]: return palette[2]
        if v <= qs[4]: return palette[3]
        if v <= qs[5]: return palette[4]
        return palette[-1]

    return color_func, qs


def _add_quantile_legend(
    fmap,  # folium.Map
    breaks: np.ndarray,
    palette: list[str] | None = None,
    title: str = "Total de eventos (cuantiles)",
) -> None:
    """
    Inserta una leyenda HTML fija (simple) con los cortes de cuantiles.
    """
    if palette is None:
        palette = ["#f1eef6", "#bdc9e1", "#74a9cf", "#2b8cbe", "#045a8d", "#023858"]

    labels = []
    for i in range(len(breaks) - 1):
        a, b = int(breaks[i]), int(breaks[i+1])
        labels.append((palette[i if i < len(palette) else -1], f"{a}–{b}"))

    html = f"""
    <div style="position: fixed; 
                bottom: 20px; left: 20px; z-index: 9999; 
                background-color: white; padding: 10px; border: 1px solid #ccc; 
                box-shadow: 2px 2px 2px rgba(0,0,0,0.2); font-size: 12px;">
      <div style="font-weight: bold; margin-bottom: 5px;">{title}</div>
      {"".join([f'<div style="display:flex;align-items:center;margin:2px 0;">'
                f'<div style="width:14px;height:14px;background:{c};margin-right:6px;border:1px solid #999;"></div>'
                f'<span>{lab}</span></div>' for c, lab in labels])}
    </div>
    """
    from folium import MacroElement
    from jinja2 import Template
    class Legend(MacroElement):
        def __init__(self, html):
            super().__init__()
            self._template = Template(html)
    fmap.get_root().add_child(Legend(html))




def make_folium_map_active_cells(
    active_cells_df: pd.DataFrame,
    cell_col: str = "cell_id",
    cat_col: str = "categoria_delito",
    events_col: str = "total_events",
    method: str = "h3",
    aggregate_across_categories: bool = True,
    max_cells: int = 4000,
    show_polygons: bool = True,
    tiles: str = "CartoDB positron",
    zoom_start: int = 11,
    # NUEVO: controles
    show_breakdown_popup: bool = True,
    top_n_breakdown: int = 5,
    choropleth_by_quantiles: bool = True,
    legend_title: str = "Total de eventos (cuantiles)",
) -> "folium.Map":
    """
    Mapa interactivo en Folium con celdas activas.
    - Si show_polygons=True y method='h3', dibuja hexágonos H3.
    - Si show_polygons=False, dibuja marcadores en centroides.
    - Si choropleth_by_quantiles=True, colorea por cuantiles de total_events_sum con leyenda.
    - Si show_breakdown_popup=True, agrega popup con top categorías por celda.
    """
    if folium is None:
        raise RuntimeError("folium no disponible. Instala con: pip install folium")

    # Base de centroides + totales por celda
    cent = active_cells_centroids(
        active_cells_df, cell_col=cell_col, cat_col=cat_col, events_col=events_col,
        method=method, aggregate_across_categories=aggregate_across_categories
    ).sort_values("total_events_sum", ascending=False)

    if cent.empty:
        raise ValueError("No hay celdas activas para mapear.")

    # Breakdown por categoría (top-N)
    breakdown_df = compute_cell_category_breakdown(
        active_cells_df, cell_col=cell_col, cat_col=cat_col, events_col=events_col, top_n=top_n_breakdown
    )

    cent = cent.merge(breakdown_df[[cell_col, "cat_breakdown"]], on=cell_col, how="left")

    # Centro del mapa
    ctr_lat = float(cent["lat"].median())
    ctr_lon = float(cent["lon"].median())
    m = folium.Map(location=[ctr_lat, ctr_lon], tiles=tiles, zoom_start=zoom_start)

    # Truncar para performance
    cent = cent.head(max_cells).reset_index(drop=True)

    # Color por cuantiles o por función anterior
    if choropleth_by_quantiles:
        color_func, breaks = _build_quantile_color_fn(cent["total_events_sum"], palette=None)
        _add_quantile_legend(m, breaks, palette=None, title=legend_title)
    else:
        # fallback a rampita simple si alguien desactiva choropleth
        def color_func(v): 
            return "#2b8cbe"

    for _, r in cent.iterrows():
        val = float(r["total_events_sum"])
        col = color_func(val)

        # Texto del tooltip
        ttip = (f"{r[cell_col]}<br>"
                f"eventos={int(val)} | cats={int(r['n_categories']) if pd.notna(r['n_categories']) else 'NA'}")

        # Popup con top-N categorías
        popup_html = ""
        if show_breakdown_popup:
            popup_html = _format_breakdown_html(r.get("cat_breakdown", []), max_lines=top_n_breakdown)

        if show_polygons and method == "h3":
            boundary = _h3_cell_boundary_lonlat(r[cell_col])  # [(lon,lat),...]
            poly_latlon = [(lat, lon) for (lon, lat) in boundary]  # folium espera (lat, lon)
            folium.Polygon(
                locations=poly_latlon,
                color=col, weight=1, fill=True, fill_color=col, fill_opacity=0.55,
                tooltip=ttip,
                popup=folium.Popup(popup_html, max_width=320)
            ).add_to(m)
        else:
            folium.CircleMarker(
                location=[r["lat"], r["lon"]],
                radius=6,
                color=col,
                fill=True,
                fill_opacity=0.85,
                tooltip=ttip,
                popup=folium.Popup(popup_html, max_width=320)
            ).add_to(m)

    return m




# Superposición de celdas activas + TOP-k hotspots (Folium)
def make_folium_map_active_cells_with_topk(
    active_cells_df: pd.DataFrame,
    topk_df: pd.DataFrame,
    # columnas de activos
    cell_col: str = "cell_id",
    cat_col: str = "categoria_delito",
    events_col: str = "total_events",
    # columnas de predicciones (top-k)
    pred_lat_col: str = "lat",
    pred_lon_col: str = "lon",
    pred_score_col: str = "score",
    # opciones de render
    method: str = "h3",
    aggregate_across_categories: bool = True,
    max_cells: int = 4000,
    show_polygons: bool = True,
    tiles: str = "CartoDB positron",
    zoom_start: int = 11,
    choropleth_by_quantiles: bool = True,
    legend_title: str = "Total de eventos (cuantiles)",
    # filtros/estilo
    filter_categories: Optional[List[str]] = None,  # si None, muestra todas las categorías presentes en topk_df
    hotspot_radius: int = 6,
    hotspot_color: str = "#d7301f",   # rojo ladrillo
    hotspot_fill_opacity: float = 0.85,
) -> "folium.Map":
    """
    Dibuja un mapa Folium con:
      - celdas activas (polígonos H3) como choropleth por intensidad
      - marcadores de TOP-k hotspots (topk_df) por categoría

    Notas:
      - Si 'filter_categories' no es None, sólo muestra esas categorías en la capa de hotspots.
      - 'topk_df' debe tener coordenadas de centroides en columnas pred_lat_col/pred_lon_col.
    """
    if folium is None:
        raise RuntimeError("folium no disponible. Instala con: pip install folium")

    # Base: centroides + totales por celda
    cent = active_cells_centroids(
        active_cells_df, cell_col=cell_col, cat_col=cat_col, events_col=events_col,
        method=method, aggregate_across_categories=aggregate_across_categories
    ).sort_values("total_events_sum", ascending=False)

    if cent.empty:
        raise ValueError("No hay celdas activas para mapear.")

    # Centro del mapa (mediana)
    ctr_lat = float(cent["lat"].median())
    ctr_lon = float(cent["lon"].median())
    m = folium.Map(location=[ctr_lat, ctr_lon], tiles=tiles, zoom_start=zoom_start)

    # Truncamos número de celdas por performance
    cent = cent.head(max_cells).reset_index(drop=True)

    # Choropleth por cuantiles (si procede)
    if choropleth_by_quantiles:
        color_func, breaks = _build_quantile_color_fn(cent["total_events_sum"], palette=None)
        _add_quantile_legend(m, breaks, palette=None, title=legend_title)
    else:
        def color_func(v): 
            return "#2b8cbe"

    # Capa de polígonos/centroides para celdas activas
    cell_layer = folium.FeatureGroup(name="Celdas activas", show=True)
    for _, r in cent.iterrows():
        val = float(r["total_events_sum"])
        col = color_func(val)
        ttip = (f"{r[cell_col]}<br>"
                f"eventos={int(val)} | cats={int(r['n_categories']) if pd.notna(r['n_categories']) else 'NA'}")

        if show_polygons and method == "h3":
            boundary = _h3_cell_boundary_lonlat(r[cell_col])  # [(lon,lat),...]
            poly_latlon = [(lat, lon) for (lon, lat) in boundary]  # folium usa (lat, lon)
            folium.Polygon(
                locations=poly_latlon,
                color=col, weight=1, fill=True, fill_color=col, fill_opacity=0.55,
                tooltip=ttip
            ).add_to(cell_layer)
        else:
            folium.CircleMarker(
                location=[r["lat"], r["lon"]],
                radius=5,
                color=col,
                fill=True,
                fill_opacity=0.8,
                tooltip=ttip
            ).add_to(cell_layer)
    cell_layer.add_to(m)

    # Capa de hotspots (top-k)
    if topk_df is not None and not topk_df.empty:
        preds = topk_df.copy()
        _check_cols(preds, [cat_col, pred_lat_col, pred_lon_col, pred_score_col])

        # Filtrar categorías si se indicó
        if filter_categories is not None:
            preds = preds[preds[cat_col].isin(filter_categories)].copy()

        # Coercer numéricos y omitir inválidos
        preds[pred_lat_col] = pd.to_numeric(preds[pred_lat_col], errors="coerce")
        preds[pred_lon_col] = pd.to_numeric(preds[pred_lon_col], errors="coerce")
        preds = preds[preds[[pred_lat_col, pred_lon_col]].notna().all(axis=1)]

        # Ordenar para que los puntajes más altos no queden ocultos
        preds = preds.sort_values(pred_score_col, ascending=False)

        hotspot_layer = folium.FeatureGroup(name="TOP-k hotspots", show=True)
        for _, p in preds.iterrows():
            score = float(p[pred_score_col])
            cat = str(p[cat_col])
            tooltip = f"{cat}<br>score={int(score)}"
            folium.CircleMarker(
                location=[float(p[pred_lat_col]), float(p[pred_lon_col])],
                radius=int(hotspot_radius),
                color=hotspot_color,
                fill=True,
                fill_color=hotspot_color,
                fill_opacity=hotspot_fill_opacity,
                tooltip=tooltip
            ).add_to(hotspot_layer)
        hotspot_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m








#  Fecha de evento y limpieza
def make_fecha_evento(
    df: pd.DataFrame,
    fecha_primary: str = "fecha_hecho",
    fecha_fallback: str = "fecha_inicio",
    out_col: str = "fecha_evento",
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Construye la columna 'fecha_evento' con prioridad:
      1) fecha_primary (p.ej., fecha_hecho)
      2) fecha_fallback (p.ej., fecha_inicio), si la primaria está NaT
    Si drop_na=True, elimina filas sin fecha_evento final (NaT).
    """
    _check_cols(df, [fecha_primary, fecha_fallback])
    out = df.copy()
    # asegurar tipo datetime (por si el casting no se ha aplicado en el entorno actual)
    out[fecha_primary] = pd.to_datetime(out[fecha_primary], errors="coerce")
    out[fecha_fallback] = pd.to_datetime(out[fecha_fallback], errors="coerce")
    out[out_col] = out[fecha_primary].where(out[fecha_primary].notna(), out[fecha_fallback])
    if drop_na:
        before = len(out)
        out = out[out[out_col].notna()].copy()
        removed = before - len(out)
        if removed > 0:
            print(f"[INFO] make_fecha_evento: {removed} filas sin fecha_evento fueron eliminadas.")
    # normalizar a día (sin hora) para facilitar agregación diaria
    out[out_col] = out[out_col].dt.floor("D")
    return out



def drop_duplicate_incidents(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = "first",
) -> pd.DataFrame:
    """
    Elimina posibles duplicados. Por defecto sugiere columnas robustas:
      ['latitud','longitud','fecha_evento','categoria_delito']
    Ajusta 'subset' si tienes un ID único confiable.
    """
    out = df.copy()
    if subset is None:
        needed = ["latitud", "longitud", "fecha_evento", "categoria_delito"]
        missing = [c for c in needed if c not in out.columns]
        if missing:
            raise ValueError(f"Faltan columnas para deduplicar: {missing}")
        subset = needed
    before = len(out)
    out = out.drop_duplicates(subset=subset, keep=keep)
    removed = before - len(out)
    if removed > 0:
        print(f"[INFO] drop_duplicate_incidents: {removed} duplicados removidos (subset={subset}).")
    return out



# Asegurar celdas y columnas clave
def ensure_cells_column(
    df: pd.DataFrame,
    lon_col: str = "longitud",
    lat_col: str = "latitud",
    method: str = "h3",
    h3_res: int = 8,
    geohash_prec: int = 6,
    cell_col: str = "cell_id",
) -> pd.DataFrame:
    """
    Garantiza la existencia de 'cell_id' (o el nombre que indiques en cell_col).
    Si no existe, lo crea con H3/geohash; si existe, respeta su contenido.
    """
    out = df.copy()
    if cell_col not in out.columns:
        out = index_points_to_cells(
            out, lon_col=lon_col, lat_col=lat_col,
            method=method, h3_res=h3_res, geohash_prec=geohash_prec,
            out_col=cell_col,
        )
    else:
        # coerción a object/str y respetar NaN donde no hay coordenadas válidas
        out[cell_col] = out[cell_col].astype("object")
    # filtrado suave: aún NO eliminamos NaN aquí (se hará antes de agregar)
    return out





# Agregación temporal continua por (celda, categoría, día)
def aggregate_counts_by_cell_category(
    df: pd.DataFrame,
    date_col: str = "fecha_evento",
    cell_col: str = "cell_id",
    cat_col: str = "categoria_delito",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    freq: str = "D",  # 'D' diario, 'W' semanal
    complete_grid: bool = True,
) -> pd.DataFrame:
    """
    Agrega conteos por (cell, categoría) con continuidad temporal.
    - Para 'D' usa piso diario.
    - Para 'W' (u otra frecuencia) usa pd.Grouper(key='ds', freq=freq).
    - Si complete_grid=True, reindexa cada grupo para cubrir todo el rango con y=0.
    """
    _check_cols(df, [date_col, cell_col, cat_col])
    x = df.copy()

    # Filtra filas con fecha/celda válidas
    x = x[x[date_col].notna() & x[cell_col].notna()].copy()
    x[date_col] = pd.to_datetime(x[date_col], errors="coerce")

    if start_date:
        x = x[x[date_col] >= pd.to_datetime(start_date)]
    if end_date:
        x = x[x[date_col] <= pd.to_datetime(end_date)]

    # Normalizar a día; luego agregamos según 'freq'
    x["ds"] = x[date_col].dt.floor("D")

    if freq == "D":
        g = (
            x.groupby([cell_col, cat_col, "ds"])
             .size()
             .rename("y")
             .reset_index()
        )
    else:
        # Agrupación con pd.Grouper evita el KeyError de resample(on='ds')
        g = (
            x.groupby([cell_col, cat_col, pd.Grouper(key="ds", freq=freq)])
             .size()
             .rename("y")
             .reset_index()
        )

    if not complete_grid:
        g["y"] = g["y"].astype(float)
        return g

    # complete grid: continuidad temporal por grupo
    all_groups = g[[cell_col, cat_col]].drop_duplicates().reset_index(drop=True)
    ds_min = g["ds"].min()
    ds_max = g["ds"].max()
    all_periods = pd.date_range(ds_min, ds_max, freq=freq)

    frames = []
    for _, row in all_groups.iterrows():
        cid = row[cell_col]; cat = row[cat_col]
        gi = g[(g[cell_col] == cid) & (g[cat_col] == cat)].set_index("ds")
        gi = gi.reindex(all_periods).fillna(0.0)
        gi.index.name = "ds"
        gi = gi.reset_index()
        gi[cell_col] = cid
        gi[cat_col] = cat
        frames.append(gi)

    out = pd.concat(frames, ignore_index=True)
    out["y"] = out["y"].astype(float)
    return out




# Métricas de sparsidad / cobertura QA
def sparsity_profile(
    counts_df: pd.DataFrame,
    cell_col: str = "cell_id",
    cat_col: str = "categoria_delito",
    date_col: str = "ds",
    y_col: str = "y",
) -> pd.DataFrame:
    """
    Calcula perfiles de sparsidad por (celda, categoría):
      - periods_total, periods_active, active_ratio
      - y_zero_ratio, total_events, y_p50, y_p90, y_p99
      - range_start, range_end, range_periods
    """
    _check_cols(counts_df, [cell_col, cat_col, date_col, y_col])

    def agg(g: pd.DataFrame) -> pd.Series:
        y = g[y_col].values
        n = len(y)
        active = int((y > 0).sum())
        return pd.Series({
            "periods_total": n,
            "periods_active": active,
            "active_ratio": active / max(n, 1),
            "y_zero_ratio": float((y == 0).mean()),
            "total_events": float(y.sum()),
            "y_p50": float(np.percentile(y, 50)),
            "y_p90": float(np.percentile(y, 90)),
            "y_p99": float(np.percentile(y, 99)),
            "range_start": g[date_col].min(),
            "range_end": g[date_col].max(),
        })

    prof = counts_df.groupby([cell_col, cat_col]).apply(agg).reset_index()
    prof["range_periods"] = (prof["range_end"] - prof["range_start"]).dt.days + 1
    return prof


def find_ghost_cells(
    sparsity_df: pd.DataFrame,
    min_active_ratio: float = 0.05,
    min_total_events: int = 3,
) -> pd.DataFrame:
    """
    Detecta 'celdas fantasma': poca continuidad o muy baja señal.
    Criterio por defecto:
      - active_ratio < 0.05 (5% de periodos con actividad)
      - o total_events < 3
    """
    _check_cols(sparsity_df, ["cell_id", "categoria_delito", "active_ratio", "total_events"])
    mask = (sparsity_df["active_ratio"] < float(min_active_ratio)) | (sparsity_df["total_events"] < float(min_total_events))
    ghosts = sparsity_df[mask].copy()
    return ghosts





# Contexto de clusters para features 
def build_cell_to_cluster_map(
    df_events: pd.DataFrame,
    cell_col: str = "cell_id",
    cluster_col: str = "cluster_id",
    min_votes: int = 1,
) -> pd.DataFrame:
    """
    Asigna a cada 'cell_id' un 'cluster_id' contextual por mayoría (mode),
    usando el historial de incidentes:
      - Cuenta ocurrencias por (cell_id, cluster_id)
      - Toma el cluster con mayor conteo para esa celda
      - Filtra celdas con muy pocos votos si min_votes > 1
    Devuelve: DataFrame [cell_id, cluster_id, votes]
    """
    _check_cols(df_events, [cell_col, cluster_col])
    g = (df_events.dropna(subset=[cell_col, cluster_col])
                  .groupby([cell_col, cluster_col])
                  .size()
                  .rename("votes")
                  .reset_index())
    # elegir cluster con más votos por celda
    idx = g.groupby(cell_col)["votes"].idxmax()
    best = g.loc[idx].reset_index(drop=True)
    if min_votes > 1:
        best = best[best["votes"] >= int(min_votes)].reset_index(drop=True)
    return best  # [cell_id, cluster_id, votes]


def attach_cluster_context(
    counts_df: pd.DataFrame,
    cell_to_cluster_df: pd.DataFrame,
    cell_col: str = "cell_id",
    cluster_col: str = "cluster_id",
) -> pd.DataFrame:
    """
    Enriquecer el counts_df con el cluster contextual por celda.
    """
    _check_cols(counts_df, [cell_col])
    _check_cols(cell_to_cluster_df, [cell_col, cluster_col])
    out = counts_df.merge(cell_to_cluster_df[[cell_col, cluster_col]], on=cell_col, how="left")
    return out




def _ghost_profile_to_events_like(
    ghosts_profile_df: pd.DataFrame,
    events_col_out: str = "total_events",
) -> pd.DataFrame:
    """
    Convierte el perfil de sparsidad de fantasmas (sparsity_df filtrado)
    a un DF 'tipo eventos' mínimo para mapear:
        [cell_id, events_col_out]
    Suma total_events por celda. No requiere categoria_delito.
    """
    _check_cols(ghosts_profile_df, ["cell_id", "total_events"])
    g = (ghosts_profile_df.groupby("cell_id")["total_events"]
                           .sum()
                           .rename(events_col_out)
                           .reset_index())
    return g


def make_folium_map_active_vs_ghost(
    active_cells_df: pd.DataFrame,     # típico: [cell_id, categoria_delito, total_events]
    ghosts_profile_df: pd.DataFrame,   # típico: salida de find_ghost_cells(...) (subset de sparsity_profile)
    cell_col: str = "cell_id",
    cat_col: str = "categoria_delito",
    events_col: str = "total_events",
    method: str = "h3",
    aggregate_across_categories: bool = True,
    max_active_cells: int = 4000,
    max_ghost_cells: int = 4000,
    show_polygons: bool = True,
    tiles: str = "CartoDB positron",
    zoom_start: int = 11,
    # estilos
    active_legend_title: str = "Celdas activas: total de eventos (cuantiles)",
    ghosts_color: str = "#d7301f",      # rojo ladrillo
    ghosts_fill_opacity: float = 0.35,
    ghosts_edge_weight: int = 1,
) -> "folium.Map":
    """
    Mapa Folium con dos capas:
      - Celdas ACTIVAS: choropleth por cuantiles (intensidad) con leyenda.
      - Celdas FANTASMA: capa aparte, color fijo distinguible.

    Recomendado cuando 'ghosts_profile_df' viene de `find_ghost_cells(...)`.
    """
    if folium is None:
        raise RuntimeError("folium no disponible. Instala con: pip install folium")

    # --- Activas (agregado y centroides)
    act_cent = active_cells_centroids(
        active_cells_df, cell_col=cell_col, cat_col=cat_col, events_col=events_col,
        method=method, aggregate_across_categories=aggregate_across_categories
    ).sort_values("total_events_sum", ascending=False)

    if act_cent.empty:
        raise ValueError("No hay celdas activas para mapear.")

    # Centro del mapa por mediana de activas
    ctr_lat = float(act_cent["lat"].median())
    ctr_lon = float(act_cent["lon"].median())
    m = folium.Map(location=[ctr_lat, ctr_lon], tiles=tiles, zoom_start=zoom_start)

    # Limitar cantidad por performance
    act_cent = act_cent.head(max_active_cells).reset_index(drop=True)

    # Choropleth por cuantiles para activas
    color_func, breaks = _build_quantile_color_fn(act_cent["total_events_sum"], palette=None)
    _add_quantile_legend(m, breaks, palette=None, title=active_legend_title)

    # Capa ACTIVAS
    layer_active = folium.FeatureGroup(name="Celdas ACTIVAS", show=True)
    for _, r in act_cent.iterrows():
        val = float(r["total_events_sum"])
        col = color_func(val)
        ttip = (f"{r[cell_col]}<br>"
                f"eventos={int(val)} | cats={int(r['n_categories']) if pd.notna(r['n_categories']) else 'NA'}")
        if show_polygons and method == "h3":
            boundary = _h3_cell_boundary_lonlat(r[cell_col])  # [(lon,lat),...]
            poly_latlon = [(lat, lon) for (lon, lat) in boundary]
            folium.Polygon(
                locations=poly_latlon,
                color=col, weight=1, fill=True, fill_color=col, fill_opacity=0.55,
                tooltip=ttip
            ).add_to(layer_active)
        else:
            folium.CircleMarker(
                location=[r["lat"], r["lon"]],
                radius=5, color=col, fill=True, fill_opacity=0.85, tooltip=ttip
            ).add_to(layer_active)
    layer_active.add_to(m)

    # --- Fantasmas (convertir perfil a 'tipo eventos' y centroides)
    ghosts_events_like = _ghost_profile_to_events_like(ghosts_profile_df, events_col_out=events_col)
    if ghosts_events_like.empty:
        folium.LayerControl(collapsed=False).add_to(m)
        return m

    ghost_cent = active_cells_centroids(
        ghosts_events_like, cell_col=cell_col, cat_col=cat_col, events_col=events_col,
        method=method, aggregate_across_categories=True  # no tenemos desglose por categoría aquí
    ).sort_values("total_events_sum", ascending=False).head(max_ghost_cells).reset_index(drop=True)

    # Capa FANTASMAS
    layer_ghosts = folium.FeatureGroup(name="Celdas FANTASMA", show=True)
    for _, r in ghost_cent.iterrows():
        ttip = f"{r[cell_col]}<br>fantasma | eventos={int(r['total_events_sum'])}"
        if show_polygons and method == "h3":
            boundary = _h3_cell_boundary_lonlat(r[cell_col])
            poly_latlon = [(lat, lon) for (lon, lat) in boundary]
            folium.Polygon(
                locations=poly_latlon,
                color=ghosts_color, weight=ghosts_edge_weight,
                fill=True, fill_color=ghosts_color, fill_opacity=ghosts_fill_opacity,
                tooltip=ttip
            ).add_to(layer_ghosts)
        else:
            folium.CircleMarker(
                location=[r["lat"], r["lon"]],
                radius=5,
                color=ghosts_color,
                fill=True, fill_color=ghosts_color, fill_opacity=min(ghosts_fill_opacity + 0.2, 1.0),
                tooltip=ttip
            ).add_to(layer_ghosts)
    layer_ghosts.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m



def ghost_proximity_to_active(
    active_cells_df: pd.DataFrame,
    ghosts_profile_df: pd.DataFrame,
    cell_col: str = "cell_id",
    cat_col: str = "categoria_delito",
    events_col: str = "total_events",
    method: str = "h3",
    max_active: int = 5000,
    max_ghosts: int = 5000,
    radius_m: float = 1000.0,
) -> Dict[str, float]:
    """
    Calcula métricas de proximidad espacial entre celdas fantasma y activas:
      - % de celdas fantasma dentro de 'radius_m' metros de alguna activa
      - distancia media mínima y p50/p90/p99
    Basado en los centroides de cada celda.
    """
    _check_cols(active_cells_df, [cell_col])
    _check_cols(ghosts_profile_df, [cell_col, "total_events"])

    # Centroides (lat/lon) de activas y fantasmas
    act_cent = active_cells_centroids(
        active_cells_df, cell_col=cell_col, cat_col=cat_col, events_col=events_col,
        method=method, aggregate_across_categories=True
    ).head(max_active)
    ghost_cent = active_cells_centroids(
        _ghost_profile_to_events_like(ghosts_profile_df, events_col_out=events_col),
        cell_col=cell_col, cat_col=cat_col, events_col=events_col,
        method=method, aggregate_across_categories=True
    ).head(max_ghosts)

    act_pts = act_cent[["lat", "lon"]].to_numpy()
    ghost_pts = ghost_cent[["lat", "lon"]].to_numpy()

    if len(act_pts) == 0 or len(ghost_pts) == 0:
        return {"n_active": len(act_pts), "n_ghosts": len(ghost_pts), "pct_nearby": np.nan}

    dists_min = []
    for lat_g, lon_g in ghost_pts:
        dmin = np.inf
        for lat_a, lon_a in act_pts:
            d = haversine_distance_m(lat_g, lon_g, lat_a, lon_a)
            if d < dmin:
                dmin = d
                if dmin <= radius_m:  # early exit
                    break
        dists_min.append(dmin)

    dists = np.array(dists_min)
    pct_nearby = float((dists <= radius_m).mean() * 100)
    return {
        "n_active": len(act_pts),
        "n_ghosts": len(ghost_pts),
        "radius_m": radius_m,
        "pct_nearby": round(pct_nearby, 2),
        "mean_min_dist_m": float(np.mean(dists)),
        "p50_m": float(np.percentile(dists, 50)),
        "p90_m": float(np.percentile(dists, 90)),
        "p99_m": float(np.percentile(dists, 99)),
    }




def compute_ghosts_min_distance_to_active(
    active_cells_df: pd.DataFrame,      # típicamente: [cell_id, categoria_delito, total_events]
    ghosts_profile_df: pd.DataFrame,    # salida de find_ghost_cells(..) (subset de sparsity_profile)
    cell_col: str = "cell_id",
    cat_col: str = "categoria_delito",
    events_col: str = "total_events",
    method: str = "h3",
    max_active: int = 10000,
    max_ghosts: int = 10000,
    radius_m: float = 1000.0,
) -> pd.DataFrame:
    """
    Para cada celda fantasma, calcula la distancia mínima (metros) al centroide
    de cualquier celda activa. Devuelve un DataFrame por 'ghost_cell_id' con:
        - ghost_cell_id
        - min_dist_m
        - nearest_active_cell
        - is_near_active (min_dist_m <= radius_m)
        - (opcional) ghost_total_events, ghost_n_categories (si están disponibles)
    """
    _check_cols(ghosts_profile_df, [cell_col, "total_events"])
    _check_cols(active_cells_df, [cell_col])

    # Centroides de ACTIVAS (agregadas por celda)
    act_cent = active_cells_centroids(
        active_cells_df,
        cell_col=cell_col, cat_col=cat_col, events_col=events_col,
        method=method, aggregate_across_categories=True
    ).head(max_active)

    # Centroides de FANTASMAS (convertir perfil a "events-like" por celda)
    ghosts_events_like = _ghost_profile_to_events_like(ghosts_profile_df, events_col_out=events_col)
    ghost_cent = active_cells_centroids(
        ghosts_events_like,
        cell_col=cell_col, cat_col=cat_col, events_col=events_col,
        method=method, aggregate_across_categories=True
    ).head(max_ghosts)

    if act_cent.empty or ghost_cent.empty:
        return pd.DataFrame(columns=["ghost_cell_id", "min_dist_m", "nearest_active_cell", "is_near_active"])

    # Matrices de coordenadas (lat, lon)
    A = act_cent[["lat", "lon"]].to_numpy()        # (Na, 2)
    G = ghost_cent[["lat", "lon"]].to_numpy()      # (Ng, 2)

    # Cálculo de distancias mínima por fantasma (Ng x Na) — bucle doble (dimensiones moderadas)
    # Si prefieres, puedes vectorizar; con ~1e6 pares esto es suficientemente rápido.
    min_d = []
    nearest_idx = []
    for i in range(G.shape[0]):
        lat_g, lon_g = float(G[i, 0]), float(G[i, 1])
        d_best = np.inf
        j_best = -1
        for j in range(A.shape[0]):
            lat_a, lon_a = float(A[j, 0]), float(A[j, 1])
            d = haversine_distance_m(lat_g, lon_g, lat_a, lon_a)
            if d < d_best:
                d_best = d
                j_best = j
        min_d.append(d_best)
        nearest_idx.append(j_best)

    res = pd.DataFrame({
        "ghost_cell_id": ghost_cent[cell_col].values,
        "min_dist_m": np.array(min_d, dtype=float),
        "nearest_active_cell": act_cent.iloc[nearest_idx][cell_col].values,
    })
    res["is_near_active"] = res["min_dist_m"] <= float(radius_m)

    # (Opcional) Adjuntar info de fantasma: total de eventos (desde ghosts_profile_df)
    ghosts_tot = (ghosts_profile_df.groupby(cell_col)["total_events"]
                                 .sum()
                                 .rename("ghost_total_events")
                                 .reset_index())
    res = res.merge(ghosts_tot.rename(columns={cell_col: "ghost_cell_id"}), on="ghost_cell_id", how="left")

    # (Opcional) Si tu ghosts_profile_df tuviera número de categorías por celda, puedes anexarlo aquí.
    return res


def filter_ghosts_by_distance(
    ghosts_feat_df: pd.DataFrame,
    dist_threshold_m: float = 2000.0
) -> pd.DataFrame:
    """
    Filtra celdas fantasma con distancia mínima superior a 'dist_threshold_m'.
    Retorna subset (ghosts_far).
    """
    _check_cols(ghosts_feat_df, ["ghost_cell_id", "min_dist_m"])
    return ghosts_feat_df[ghosts_feat_df["min_dist_m"] > float(dist_threshold_m)].copy()





# Asignar cluster a PUNTOS
def assign_geo_clusters_to_points(
    model,
    df_points: pd.DataFrame,
    lon_col: str = "longitud",
    lat_col: str = "latitud",
    out_cluster_col: str = "cluster_id",
    out_dist_m_col: str = "cluster_dist_m",
    centroids_df: Optional[pd.DataFrame] = None,   # requiere ['cluster_id','lon','lat'] si se usa fallback
) -> pd.DataFrame:
    """
    Asigna cluster geográfico a puntos (incidentes u observaciones con lat/lon).
    - Si 'model' tiene .predict: usa [lon, lat] (o pipeline que encapsule scaler).
    - Si no hay modelo o falla: fallback por centroide más cercano (Haversine) con 'centroids_df'.

    Retorna df con columnas nuevas:
        out_cluster_col, out_dist_m_col
    """
    out = df_points.copy()
    # coerciones numéricas
    out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce")
    out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce")
    valid = out[lon_col].notna() & out[lat_col].notna()

    out[out_cluster_col] = pd.NA
    out[out_dist_m_col] = np.nan

    used_model = False
    if model is not None:
        try:
            X = out.loc[valid, [lon_col, lat_col]].to_numpy(dtype=float)
            labels = model.predict(X)
            out.loc[valid, out_cluster_col] = labels
            used_model = True
        except Exception as e:
            print(f"[WARN] assign_geo_clusters_to_points: predict() falló, uso fallback por centroides. Detalle: {e}")

    # Fallback por centroides (Haversine)
    if not used_model:
        if centroids_df is None or centroids_df.empty:
            raise RuntimeError("assign_geo_clusters_to_points: no hay modelo ni centroides para fallback.")
        C = centroids_df[["cluster_id", "lon", "lat"]].copy()
        C["lon"] = pd.to_numeric(C["lon"], errors="coerce")
        C["lat"] = pd.to_numeric(C["lat"], errors="coerce")
        C = C.dropna(subset=["lon", "lat"])
        C_np = C[["lat", "lon"]].to_numpy(dtype=float)

        for idx in out.index[valid]:
            lat, lon = float(out.at[idx, lat_col]), float(out.at[idx, lon_col])
            best_d, best_c = np.inf, None
            for j in range(C_np.shape[0]):
                d = haversine_distance_m(lat, lon, C_np[j, 0], C_np[j, 1])
                if d < best_d:
                    best_d = d
                    best_c = C.iloc[j]["cluster_id"]
            out.at[idx, out_cluster_col] = best_c
            out.at[idx, out_dist_m_col] = float(best_d)

    # Si usamos modelo y tenemos centroides, calculamos distancia al centroide asignado
    if used_model and centroids_df is not None and not centroids_df.empty:
        C = centroids_df[["cluster_id", "lon", "lat"]].dropna()
        cmap = {
            int(row["cluster_id"]): (float(row["lat"]), float(row["lon"]))
            for _, row in C.iterrows() if pd.notna(row["cluster_id"])
        }
        for idx in out.index[valid]:
            cid = out.at[idx, out_cluster_col]
            if pd.notna(cid) and int(cid) in cmap:
                lat, lon = float(out.at[idx, lat_col]), float(out.at[idx, lon_col])
                clat, clon = cmap[int(cid)]
                out.at[idx, out_dist_m_col] = haversine_distance_m(lat, lon, clat, clon)

    return out



# Asignar cluster a CELDAS (por centroide H3)
def assign_geo_clusters_to_cells_by_centroid(
    cells_df: pd.DataFrame,
    method: str = "h3",
    cell_col: str = "cell_id",
    out_cluster_col: str = "cluster_id",
    out_dist_m_col: str = "cluster_dist_m",
    model=None,
    centroids_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Asigna cluster contextual a cada celda H3 por su centroide (lon/lat).
    - Usa assign_geo_clusters_to_points internamente.
    Retorna DF con: [cell_id, lon, lat, cluster_id, cluster_dist_m]
    """
    _check_cols(cells_df, [cell_col])
    cent = cells_to_centroids(cells_df[cell_col].dropna().unique().tolist(), method=method)  # [cell_id, lon, lat]
    cent = assign_geo_clusters_to_points(
        model=model,
        df_points=cent,
        lon_col="lon",
        lat_col="lat",
        out_cluster_col=out_cluster_col,
        out_dist_m_col=out_dist_m_col,
        centroids_df=centroids_df,
    )
    return cent



# Mapa celda->cluster por mayoría (desde incidentes)
def build_cell_to_cluster_map_from_incidents(
    df_events: pd.DataFrame,
    cell_col: str = "cell_id",
    cluster_col: str = "cluster_id",
    min_votes: int = 1,
) -> pd.DataFrame:
    """
    Construye mapeo celda->cluster por mayoría de incidentes etiquetados.
    Retorna: [cell_id, cluster_id, votes]
    """
    _check_cols(df_events, [cell_col, cluster_col])
    g = (
        df_events.dropna(subset=[cell_col, cluster_col])
                 .groupby([cell_col, cluster_col])
                 .size().rename("votes").reset_index()
    )
    if g.empty:
        return g
    idx = g.groupby(cell_col)["votes"].idxmax()
    best = g.loc[idx].reset_index(drop=True)
    if min_votes > 1:
        best = best[best["votes"] >= int(min_votes)].reset_index(drop=True)
    return best[["cell_id", "cluster_id", "votes"]]



# Adjuntar cluster a series agregadas
def attach_cluster_context(
    counts_df: pd.DataFrame,
    cell_to_cluster_df: pd.DataFrame,
    cell_col: str = "cell_id",
    cluster_col: str = "cluster_id",
) -> pd.DataFrame:
    """
    Enriquecer counts_df con cluster contextual por celda.
    """
    _check_cols(counts_df, [cell_col])
    _check_cols(cell_to_cluster_df, [cell_col, cluster_col])
    out = counts_df.merge(
        cell_to_cluster_df[[cell_col, cluster_col]],
        on=cell_col, how="left"
    )
    return out




# Merge de features de ghosts a las series
def merge_ghosts_features(
    counts_df: pd.DataFrame,
    ghosts_feat_df: pd.DataFrame,
    cell_col: str = "cell_id",
    ghost_cell_col: str = "ghost_cell_id",
) -> pd.DataFrame:
    """
    Une features de 'ghosts_feat' (min_dist y flag) por celda:
      - ghosts_feat_df espera columnas: [ghost_cell_id, ghost_min_dist_m, ghost_is_near_active]
    """
    _check_cols(counts_df, [cell_col])
    needed = [ghost_cell_col, "ghost_min_dist_m", "ghost_is_near_active"]
    _check_cols(ghosts_feat_df, needed)
    feat = ghosts_feat_df.rename(columns={
        ghost_cell_col: cell_col
    })[[cell_col, "ghost_min_dist_m", "ghost_is_near_active"]].drop_duplicates(subset=[cell_col])
    out = counts_df.merge(feat, on=cell_col, how="left")
    return out





def aggregate_cluster_density(
    df: pd.DataFrame,
    date_col: str = "ds",
    cluster_col: str = "cluster_id",
    y_col: str = "y",
    freq: str = "W",
) -> pd.DataFrame:
    """
    Suma de y por (cluster_id, periodo), con agrupación temporal segura.
    Devuelve columnas: [cluster_id, ds, cluster_y]
    """
    _check_cols(df, [date_col, cluster_col, y_col])
    x = df.copy()
    x[date_col] = pd.to_datetime(x[date_col], errors="coerce")

    if not freq or freq == "D":
        g = (
            x.groupby([cluster_col, date_col])[y_col]
             .sum()
             .rename("cluster_y")
             .reset_index()
        )
    else:
        g = (
            x.groupby([cluster_col, pd.Grouper(key=date_col, freq=freq)])[y_col]
             .sum()
             .rename("cluster_y")
             .reset_index()
        )
    return g




def assign_clusters_to_cells_by_centroid(
    cells_df: pd.DataFrame,
    method: str = "h3",
    cell_col: str = "cell_id",
    out_cluster_col: str = "cluster_id",
    out_dist_m_col: str = "cluster_dist_m",
    model=None,
    centroids_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Asigna cluster contextual a cada celda, usando su centroide:
      - Si hay 'model', intenta predecir con [lon, lat].
      - Si no, usa el centroide más cercano desde centroids_df.
    """
    _check_cols(cells_df, [cell_col])
    cent = cells_to_centroids(cells_df[cell_col].dropna().unique().tolist(), method=method)  # [cell_id, lon, lat]
    cent = assign_clusters_to_new_points(
        model=model,
        df_points=cent,
        lon_col="lon",
        lat_col="lat",
        out_cluster_col=out_cluster_col,
        out_dist_m_col=out_dist_m_col,
        centroids_df=centroids_df,
        )
    return cent  # [cell_id, lon, lat, cluster_id, cluster_dist_m]







def make_window_labels_N(
    df: pd.DataFrame,
    date_col: str = "ds",
    cell_col: str = "cell_id",
    cat_col: str = "categoria_delito",
    y_col: str = "y",
    N: int = 3,
    out_bin_col: Optional[str] = None,
    out_cnt_col: Optional[str] = None,
    keep_partial_tail: bool = False,
) -> pd.DataFrame:
    """
    Construye etiquetas FUTURAS de ventana N días por (cell, categoría) alineadas en t:
      y_cnt_N(t) = sum_{d=t..t+N-1} y_d
      y_bin_N(t) = 1[y_cnt_N(t) >= 1]

    Importante:
      - Usa 'shift(-(N-1)).rolling(N)' para sumar futuro, alineando la suma en t.
      - Las últimas (N-1) filas de cada grupo NO tienen etiqueta completa (NaN).
        * keep_partial_tail=False -> se mantienen como NaN (útil para splits)
        * keep_partial_tail=True  -> se rellenan con 0 (no recomendado para evaluación)

    Retorna:
      df con 2 columnas nuevas: out_cnt_col, out_bin_col
    """
    assert N >= 2, "N debe ser >= 2"
    if out_cnt_col is None:
        out_cnt_col = f"y_cnt_N{N}"
    if out_bin_col is None:
        out_bin_col = f"y_bin_N{N}"

    _check_cols(df, [date_col, cell_col, cat_col, y_col])
    x = df.copy()
    x[date_col] = pd.to_datetime(x[date_col], errors="coerce")

    # Asegurar orden temporal dentro de cada grupo
    x = x.sort_values([cell_col, cat_col, date_col], kind="mergesort")

    # Cálculo de suma futura alineada en t
    def _fwd_sum(g: pd.DataFrame) -> pd.Series:
        y = g[y_col].astype(float)
        # sum_{t..t+N-1} = shift(-(N-1)).rolling(window=N).sum()
        s = y.shift(-(N-1)).rolling(window=N, min_periods=N).sum()
        return s

    x[out_cnt_col] = (x.groupby([cell_col, cat_col], sort=False, group_keys=False).apply(_fwd_sum))

    if keep_partial_tail:
        x[out_cnt_col] = x[out_cnt_col].fillna(0.0)

    x[out_bin_col] = (x[out_cnt_col] >= 1.0).astype("Int64")  # etiqueta binaria

    return x


def make_temporal_splits(
    df: pd.DataFrame,
    date_col: str = "ds",
    N: int = 3,
    train_range: Tuple[str, str] = ("2016-01-01", "2022-12-31"),
    test_range: Tuple[str, str] = ("2023-01-01", "2024-12-31"),
    val_range: Optional[Tuple[str, str]] = None,
    label_cols: Optional[Tuple[str, str]] = None,  # (out_cnt_col, out_bin_col) para chequear NaNs
) -> pd.DataFrame:
    """
    Asigna 'set'={train,val,test} en función de rangos de fecha sin fuga temporal:
      - t pertenece a un split si:
          start_split <= t <= end_split
        y además:
          t + (N-1) <= end_split   (para garantizar disponibilidad de etiqueta futura)

    Si val_range es None, sólo asigna train/test.

    Si label_cols se provee, se filtran filas con etiquetas NaN dentro de cada split.

    Retorna:
      df con columna 'set' (string) y 'set' NaN para filas fuera de cualquier split.
    """
    _check_cols(df, [date_col])
    x = df.copy()
    x[date_col] = pd.to_datetime(x[date_col], errors="coerce")

    # Pre-calcular t_end_label = t + (N-1) días
    x["_t_end_label"] = x[date_col] + pd.to_timedelta(N - 1, unit="D")

    # Construir máscara por split respetando el fin para la etiqueta
    tr_start, tr_end = pd.to_datetime(train_range[0]), pd.to_datetime(train_range[1])
    te_start, te_end = pd.to_datetime(test_range[0]), pd.to_datetime(test_range[1])
    va_start, va_end = (pd.to_datetime(val_range[0]), pd.to_datetime(val_range[1])) if val_range else (None, None)

    mask_train = (x[date_col] >= tr_start) & (x[date_col] <= tr_end) & (x["_t_end_label"] <= tr_end)
    mask_test  = (x[date_col] >= te_start) & (x[date_col] <= te_end) & (x["_t_end_label"] <= te_end)
    if val_range:
        mask_val = (x[date_col] >= va_start) & (x[date_col] <= va_end) & (x["_t_end_label"] <= va_end)
    else:
        mask_val = pd.Series(False, index=x.index)

    x["set"] = pd.NA
    x.loc[mask_train, "set"] = "train"
    if val_range:
        x.loc[mask_val & (~mask_train), "set"] = "val"
    x.loc[mask_test & (~mask_train) & (~mask_val), "set"] = "test"

    # Opcional: eliminar filas sin etiquetas dentro de cada split (NaN por falta de futuro)
    if label_cols is not None:
        cnt_col, bin_col = label_cols
        if cnt_col in x.columns:
            for split_name in ["train", "val", "test"]:
                if x["set"].eq(split_name).any():
                    idx = x["set"].eq(split_name) & x[cnt_col].isna()
                    x.loc[idx, "set"] = pd.NA  # fuera del split por no tener etiqueta

    # Limpieza
    x = x.drop(columns=["_t_end_label"])
    return x


def summarize_split_prevalence(
    df: pd.DataFrame,
    split_col: str = "set",
    bin_label_col: str = "y_bin",
    cat_col: str = "categoria_delito",
) -> pd.DataFrame:
    """
    Resumen de prevalencias por split y categoría:
      - n, n_pos, pos_rate
    """
    _check_cols(df, [split_col, bin_label_col, cat_col])
    sub = df.dropna(subset=[split_col, bin_label_col])
    g = (sub.groupby([split_col, cat_col])[bin_label_col]
            .agg(n="size", n_pos="sum")
            .reset_index())
    g["pos_rate"] = g["n_pos"] / g["n"]
    return g






def plot_train_test_series_cell_cat(
    df_split: pd.DataFrame,
    cell_id: str,
    categoria: str,
    date_col: str = "ds",
    y_col: str = "y",
    set_col: str = "set",
    freq: str = "W",           # 'W' = semanal
    title_prefix: str = "Serie semanal —",
    test_start: str = "2023-01-01",
):
    sub = df_split[(df_split["cell_id"] == cell_id) & (df_split["categoria_delito"] == categoria)].copy()
    if sub.empty:
        print("[WARN] No hay datos para ese (cell, categoría).")
        return

    sub = sub.sort_values(date_col)
    # Agregado semanal
    subW = sub.set_index(date_col).groupby(set_col).resample(freq)[y_col].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10,4))
    for split_name, g in subW.groupby(set_col):
        ax.plot(g[date_col], g[y_col], label=split_name)

    ax.axvline(pd.to_datetime(test_start), linestyle="--", linewidth=1)
    ax.set_title(f"{title_prefix} {categoria}\ncell_id={cell_id}")
    ax.set_xlabel("Fecha (semana)")
    ax.set_ylabel(f"Suma {y_col} ({freq})")
    ax.grid(alpha=0.2)
    ax.legend(title="Split")
    plt.tight_layout()
    plt.show()







def plot_train_test_series_by_category_global(
    df_split: pd.DataFrame,
    categoria: str,
    date_col: str = "ds",
    y_col: str = "y",
    set_col: str = "set",
    freq: str = "W",
    test_start: str = "2023-01-01",
):
    sub = df_split[df_split["categoria_delito"] == categoria].copy()
    if sub.empty:
        print("[WARN] No hay datos para esa categoría.")
        return
    sub = sub.sort_values(date_col)

    # Agregado semanal global (sum across all cells)
    subW = sub.set_index(date_col).groupby(set_col).resample(freq)[y_col].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10,4))
    for split_name, g in subW.groupby(set_col):
        ax.plot(g[date_col], g[y_col], label=split_name)

    ax.axvline(pd.to_datetime(test_start), linestyle="--", linewidth=1)
    ax.set_title(f"Serie semanal — Global — {categoria}")
    ax.set_xlabel("Fecha (semana)")
    ax.set_ylabel(f"Suma {y_col} ({freq}) en todas las celdas")
    ax.grid(alpha=0.2)
    ax.legend(title="Split")
    plt.tight_layout()
    plt.show()







def plot_train_test_series_by_cluster(
    df_split: pd.DataFrame,
    cluster_id: int,
    date_col: str = "ds",
    y_col: str = "y",
    set_col: str = "set",
    freq: str = "W",
    test_start: str = "2023-01-01",
    title_prefix: str = "Serie semanal por cluster —"
):
    sub = df_split[df_split["cluster_id"] == cluster_id].copy()
    if sub.empty:
        print("[WARN] No hay datos para ese cluster.")
        return
    sub = sub.sort_values(date_col)
    subW = sub.set_index(date_col).groupby(set_col).resample(freq)[y_col].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10,4))
    for split_name, g in subW.groupby(set_col):
        ax.plot(g[date_col], g[y_col], label=split_name)

    ax.axvline(pd.to_datetime(test_start), linestyle="--", linewidth=1)
    ax.set_title(f"{title_prefix} {cluster_id}")
    ax.set_xlabel("Fecha (semana)")
    ax.set_ylabel(f"Suma {y_col} ({freq}) dentro del cluster")
    ax.grid(alpha=0.2)
    ax.legend(title="Split")
    plt.tight_layout()
    plt.show()




def _ensure_datetime(df: pd.DataFrame, date_col: str):
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

def _check_cols(df: pd.DataFrame, cols: Iterable[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")

def _safe_h3_neighbors(cell: str, k: int = 1):
    """
    Retorna conjunto de vecinos H3 (incluyendo la celda) para v4 (grid_disk)
    y v3 (k_ring). Si no hay h3 disponible, retorna {cell}.
    """
    try:
        import h3
    except Exception:
        return {cell}
    # v4
    if hasattr(h3, "grid_disk"):
        return set(h3.grid_disk(cell, k))
    # v3
    if hasattr(h3, "k_ring"):
        return set(h3.k_ring(cell, k))
    return {cell}

def _make_neighbor_map(cells: Iterable[str], k: int = 1) -> Dict[str, List[str]]:
    """
    Mapea cada cell_id -> lista de celdas del k-ring (incluye la propia).
    """
    cells = list(pd.Series(cells).dropna().unique())
    nbrs = {}
    for c in cells:
        nbrs[c] = list(_safe_h3_neighbors(c, k=k))
    return nbrs




def add_calendar_features(
    df: pd.DataFrame,
    date_col: str = "ds",
    prefix: str = "cal_",
    add_weekday_onehot: bool = False
) -> pd.DataFrame:
    """
    Agrega features de calendario sin fuga (solo depende de ds):
      - weekday (0..6), is_weekend
      - codificación cíclica: dow_sin/cos, month_sin/cos
    """
    _check_cols(df, [date_col])
    out = df.copy()
    _ensure_datetime(out, date_col)

    dow = out[date_col].dt.weekday  # 0=Mon..6=Sun
    month = out[date_col].dt.month

    out[f"{prefix}dow"] = dow
    out[f"{prefix}is_weekend"] = (dow >= 5).astype("Int64")

    out[f"{prefix}dow_sin"] = np.sin(2*np.pi * dow/7.0)
    out[f"{prefix}dow_cos"] = np.cos(2*np.pi * dow/7.0)
    out[f"{prefix}mon_sin"] = np.sin(2*np.pi * (month-1)/12.0)
    out[f"{prefix}mon_cos"] = np.cos(2*np.pi * (month-1)/12.0)

    if add_weekday_onehot:
        for d in range(7):
            out[f"{prefix}dow_{d}"] = (dow == d).astype("Int64")

    return out


def add_temporal_lags_and_rollings(
    df: pd.DataFrame,
    group_cols: List[str] = ["cell_id", "categoria_delito"],
    date_col: str = "ds",
    y_col: str = "y",
    lags: List[int] = [1,2,3,7,14,28],
    roll_windows: List[int] = [7,14,28],
    add_trend14: bool = True,
) -> pd.DataFrame:
    """
    Agrega lags y estadísticas rolling de 'y' **usando historia hasta t-1**.
    - Lags: y_lag{k} = y.shift(k)
    - Rolling (sobre y_lag1): mean/std/sum de ventanas [7,14,28]
    - Trend local (14): pendiente OLS sobre y_lag1 ventana 14 (si add_trend14)
    """
    _check_cols(df, group_cols + [date_col, y_col])

    out = df.copy()
    _ensure_datetime(out, date_col)
    out = out.sort_values(group_cols + [date_col])

    # Lags
    for k in lags:
        out[f"y_lag{k}"] = (
            out.groupby(group_cols, sort=False)[y_col]
               .shift(k)
               .astype(float)
        )

    # Rolling stats (sobre lag1 para garantizar corte en t-1)
    base = out.groupby(group_cols, sort=False)["y_lag1"]
    for w in roll_windows:
        out[f"roll{w}_mean"] = base.rolling(w, min_periods=1).mean().reset_index(level=group_cols, drop=True)
        out[f"roll{w}_std"]  = base.rolling(w, min_periods=1).std().reset_index(level=group_cols, drop=True)
        out[f"roll{w}_sum"]  = base.rolling(w, min_periods=1).sum().reset_index(level=group_cols, drop=True)

    # Trend 14 días sobre y_lag1 (OLS pendiente)
    if add_trend14 and (14 in roll_windows or True):
        def _slope14(s: pd.Series) -> pd.Series:
            # pendiente en ventana móvil sobre índices 0..13
            x = np.arange(14, dtype=float)
            # función rolling apply devuelve NaN si len<14
            def slope_win(win):
                if np.isnan(win).any():
                    # si hay NaN dentro de ventana, ignoramos (NaN)
                    if np.sum(~np.isnan(win)) < 2:
                        return np.nan
                    # tomar solo válidos
                    xv = x[:len(win)][~np.isnan(win)]
                    yv = win[~np.isnan(win)]
                    if len(yv) < 2:
                        return np.nan
                    A = np.vstack([xv, np.ones_like(xv)]).T
                    m, _ = np.linalg.lstsq(A, yv, rcond=None)[0]
                    return m
                A = np.vstack([x, np.ones_like(x)]).T
                m, _ = np.linalg.lstsq(A, win, rcond=None)[0]
                return m

            return s.rolling(14, min_periods=14).apply(slope_win, raw=True)

        out["trend14_slope"] = (
            out.groupby(group_cols, sort=False)["y_lag1"]
               .apply(_slope14)
               .reset_index(level=group_cols, drop=True)
        )

    return out



def add_cluster_signals(
    df: pd.DataFrame,
    cluster_col: str = "cluster_id",
    date_col: str = "ds",
    y_col: str = "y",
    lags: List[int] = [1,7],
    roll_windows: List[int] = [7,28],
    normalize_by_cells: bool = True
) -> pd.DataFrame:
    """
    Agrega features de señal regional a nivel cluster:
      - cluster_y_lag{k}: suma diaria en cluster en t-k
      - cluster_roll{w}_mean: media de cluster en ventana w sobre lag1
      - (opcional) normaliza por número de celdas del cluster
    """
    _check_cols(df, [cluster_col, date_col, y_col])

    out = df.copy()
    _ensure_datetime(out, date_col)
    out = out.sort_values([cluster_col, date_col])

    # total diario por cluster
    clus_daily = (
        out.groupby([cluster_col, date_col])[y_col].sum()
           .rename("cluster_y")
           .reset_index()
           .sort_values([cluster_col, date_col])
    )

    # Añadir lags al nivel cluster y propagar a filas
    for k in lags:
        clus_daily[f"cluster_y_lag{k}"] = (
            clus_daily.groupby(cluster_col, sort=False)["cluster_y"].shift(k)
        )

    # Rolling sobre cluster_y_lag1
    base = clus_daily.groupby(cluster_col, sort=False)["cluster_y_lag1"]
    for w in roll_windows:
        clus_daily[f"cluster_roll{w}_mean"] = base.rolling(w, min_periods=1).mean().reset_index(level=cluster_col, drop=True)

    # Normalización por #celdas del cluster (estimado a partir de df)
    if normalize_by_cells:
        n_cells = (
            out.dropna(subset=[cluster_col, "cell_id"])
               .groupby(cluster_col)["cell_id"].nunique()
               .rename("n_cells")
               .reset_index()
        )
        clus_daily = clus_daily.merge(n_cells, on=cluster_col, how="left")
        for col in [c for c in clus_daily.columns if c.startswith("cluster_y_lag") or c.startswith("cluster_roll")]:
            clus_daily[f"{col}_norm"] = clus_daily[col] / clus_daily["n_cells"].clip(lower=1)

    # Merge back
    out = out.merge(clus_daily.drop(columns=["cluster_y"]), on=[cluster_col, date_col], how="left")
    return out



def add_neighbor_aggregates(
    df: pd.DataFrame,
    cell_col: str = "cell_id",
    cat_col: str = "categoria_delito",
    date_col: str = "ds",
    y_col: str = "y",
    k: int = 1,
    from_lag: int = 1,
    to_lag: int = 7,
    agg_funcs: Tuple[str, ...] = ("mean", "p90")
) -> pd.DataFrame:
    """
    Agrega agregados de vecindad H3 (k-ring) para historia reciente:
      - Para cada t, cada celda y categoría:
        * Calcula y_lag{d} de los VECINOS (incluye celda) para d ∈ [from_lag..to_lag].
        * Aplica agregados inter-celda (mean, p90).
    Notas:
      - Requiere mapeo cell -> vecinos. Si h3 no está, degrada a solo la celda.
      - Es costoso; usar k=1 y limitar período si hace falta.
    """
    _check_cols(df, [cell_col, cat_col, date_col, y_col])
    out = df.copy()
    _ensure_datetime(out, date_col)
    out = out.sort_values([cell_col, cat_col, date_col])

    # Pre-lags por celda-categoría
    for d in range(from_lag, to_lag+1):
        out[f"y_lag{d}"] = out.groupby([cell_col, cat_col], sort=False)[y_col].shift(d).astype(float)

    # Vecinos por cell
    neighbor_map = _make_neighbor_map(out[cell_col].unique().tolist(), k=k)

    # Para eficiencia: pivot a tabla por (ds, cat, cell) -> columnas lagd
    keys = [date_col, cat_col, cell_col]
    lag_cols = [f"y_lag{d}" for d in range(from_lag, to_lag+1)]
    P = out[keys + lag_cols].copy()

    # Construimos agregados por fecha y categoría cruzando vecinos
    # (para evitar bucles, hacemos gather por fecha/cat y tomamos medias por conjunto de celdas vecino)
    def agg_for_day_cat(pdf: pd.DataFrame) -> pd.DataFrame:
        # pdf: todas las celdas de una (ds,cat)
        # index por cell_id para lookup rápido
        M = pdf.set_index(cell_col)[lag_cols]
        rows = []
        for c in M.index:
            nbrs = neighbor_map.get(c, [c])
            sub = M.loc[M.index.intersection(nbrs)]
            vals = {}
            for col in lag_cols:
                v = sub[col].values
                if "mean" in agg_funcs:
                    vals[f"nbr_{col}_mean_k{k}"] = np.nanmean(v) if len(v) else np.nan
                if "p90" in agg_funcs:
                    vals[f"nbr_{col}_p90_k{k}"] = np.nanpercentile(v, 90) if len(v) else np.nan
            r = {date_col: pdf[date_col].iloc[0], cat_col: pdf[cat_col].iloc[0], cell_col: c}
            r.update(vals)
            rows.append(r)
        return pd.DataFrame(rows)

    nbr_feats = (
        P.groupby([date_col, cat_col], sort=False, as_index=False)
         .apply(agg_for_day_cat)
         .reset_index(drop=True)
    )

    out = out.merge(nbr_feats, on=[date_col, cat_col, cell_col], how="left")
    return out




def add_long_history_priors(
    df: pd.DataFrame,
    group_cols: List[str] = ["cell_id", "categoria_delito"],
    date_col: str = "ds",
    y_col: str = "y",
    windows_days: List[int] = [365, 730],
) -> pd.DataFrame:
    """
    Priors históricos cerrados en t-1:
      - prior_mean_{w}d, prior_var_{w}d: rolling sobre y_lag1 con ventana en días 'w'
      - prior_sum_{w}d: suma histórica reciente
      - EB rate simple: (alpha + prior_sum_{365d}) / (beta + 365) con alpha=1,beta=1
    """
    _check_cols(df, group_cols + [date_col, y_col])
    out = df.copy()
    _ensure_datetime(out, date_col)
    out = out.sort_values(group_cols + [date_col])

    out["y_lag1"] = out.groupby(group_cols, sort=False)[y_col].shift(1).astype(float)

    for w in windows_days:
        base = out.groupby(group_cols, sort=False)["y_lag1"]
        out[f"prior_mean_{w}d"] = base.rolling(w, min_periods=1).mean().reset_index(level=group_cols, drop=True)
        out[f"prior_var_{w}d"]  = base.rolling(w, min_periods=2).var().reset_index(level=group_cols, drop=True)
        out[f"prior_sum_{w}d"]  = base.rolling(w, min_periods=1).sum().reset_index(level=group_cols, drop=True)

    # EB sobre 365d (ajusta si no está)
    if "prior_sum_365d" in out.columns:
        alpha, beta = 1.0, 1.0
        out["eb_rate_365d"] = (alpha + out["prior_sum_365d"]) / (beta + 365.0)

    return out



def build_features_for_window_N(
    df: pd.DataFrame,
    N: int = 3,
    date_col: str = "ds",
    cell_col: str = "cell_id",
    cat_col: str = "categoria_delito",
    y_col: str = "y",
    cluster_col: str = "cluster_id",
    include_calendar: bool = True,
    include_temporal: bool = True,
    include_neighbors: bool = True,
    include_cluster: bool = True,
    include_priors: bool = True,
    neighbor_k: int = 1,
    neighbors_lag_from: int = 1,
    neighbors_lag_to: int = 7,
    temporal_lags: List[int] = [1,2,3,7,14,28],
    temporal_rolls: List[int] = [7,14,28],
    cluster_lags: List[int] = [1,7],
    cluster_rolls: List[int] = [7,28],
    long_windows: List[int] = [365, 730],
    add_weekday_onehot: bool = False,
) -> pd.DataFrame:
    """
    Construye el dataset de features para ventana N **sin fuga**.
    - No toca/crea etiquetas; asume que y_bin_N*, y_cnt_N* ya existen.
    - Devuelve df con todas las columnas originales + nuevas features.
    """
    _check_cols(df, [date_col, cell_col, cat_col, y_col])
    X = df.copy()
    _ensure_datetime(X, date_col)

    if include_calendar:
        X = add_calendar_features(X, date_col=date_col, add_weekday_onehot=add_weekday_onehot)

    if include_temporal:
        X = add_temporal_lags_and_rollings(
            X, group_cols=[cell_col, cat_col], date_col=date_col, y_col=y_col,
            lags=temporal_lags, roll_windows=temporal_rolls, add_trend14=True
        )

    if include_cluster and (cluster_col in X.columns):
        X = add_cluster_signals(
            X, cluster_col=cluster_col, date_col=date_col, y_col=y_col,
            lags=cluster_lags, roll_windows=cluster_rolls, normalize_by_cells=True
        )

    if include_neighbors:
        X = add_neighbor_aggregates(
            X, cell_col=cell_col, cat_col=cat_col, date_col=date_col, y_col=y_col,
            k=neighbor_k, from_lag=neighbors_lag_from, to_lag=neighbors_lag_to,
            agg_funcs=("mean","p90")
        )

    if include_priors:
        X = add_long_history_priors(
            X, group_cols=[cell_col, cat_col], date_col=date_col, y_col=y_col, windows_days=long_windows
        )

    return X




def validate_no_leakage(
    df_feats: pd.DataFrame,
    date_col: str = "ds",
    feature_prefixes: Tuple[str, ...] = ("y_lag", "roll", "trend", "cluster_", "nbr_", "prior_", "cal_")
) -> pd.DataFrame:
    """
    Heurística simple: revisa que ninguna feature dependa de futuras fechas.
    - Para dos cortes aleatorios t0 < t1, se calcula un checksum de features
      con datos truncados a [<= t0] y [<= t1]; las features en t0 no deben
      cambiar si extendemos el rango a t1 (indicio de fuga).
    Retorna un pequeño reporte por prefijo.
    """
    _check_cols(df_feats, [date_col])
    dmin, dmax = pd.to_datetime(df_feats[date_col].min()), pd.to_datetime(df_feats[date_col].max())
    if pd.isna(dmin) or pd.isna(dmax) or dmin == dmax:
        return pd.DataFrame([{"check":"insuficiente_rango"}])

    rng = pd.date_range(dmin, dmax, freq="7D")
    if len(rng) < 4:
        return pd.DataFrame([{"check":"rango_corto"}])

    t0 = rng[int(len(rng)*0.4)]
    t1 = rng[int(len(rng)*0.8)]

    A = df_feats[df_feats[date_col] <= t0].copy()
    B = df_feats[df_feats[date_col] <= t1].copy()

    rep = []
    for p in feature_prefixes:
        colsA = [c for c in A.columns if c.startswith(p)]
        if not colsA:
            continue
        # checksum en A y en B recortado hasta t0
        chkA = pd.util.hash_pandas_object(A[colsA], index=False).sum()
        chkB = pd.util.hash_pandas_object(B[B[date_col] <= t0][colsA], index=False).sum()
        rep.append({"prefix": p, "stable_at_t0": bool(chkA == chkB)})

    return pd.DataFrame(rep)






def aggregate_daily_by_cluster_category(
    df: pd.DataFrame,
    date_col: str = "ds",
    y_col: str = "y",
    cluster_col: str = "cluster_id",
    cat_col: str = "categoria_delito",
) -> pd.DataFrame:
    """
    Agrega conteo diario por (cluster, categoría).
    """
    _check_cols(df, [date_col, y_col, cluster_col, cat_col])
    x = df[[date_col, y_col, cluster_col, cat_col]].copy()
    x[date_col] = pd.to_datetime(x[date_col], errors="coerce")
    out = (
        x.groupby([cluster_col, cat_col, date_col], sort=False)[y_col]
         .sum().rename("y_cluster")
         .reset_index()
         .sort_values([cluster_col, cat_col, date_col])
    )
    return out




#  Prophet (principal)
def fit_prophet_by_cluster_category(
    df_cluster_daily: pd.DataFrame,
    date_col: str = "ds",
    y_col: str = "y_cluster",
    cluster_col: str = "cluster_id",
    cat_col: str = "categoria_delito",
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = True,
    daily_seasonality: bool = False,
    changepoint_prior_scale: float = 0.1,
    holidays_df: Optional[pd.DataFrame] = None,  # opcional
) -> Dict[Tuple[int, str], object]:
    """
    Entrena Prophet por (cluster_id, categoria).
    Devuelve dict[(cluster_id, categoria)] = modelo Prophet entrenado.
    """
    try:
        from prophet import Prophet
    except Exception as e:
        raise ImportError("Instala 'prophet' (pip install prophet).") from e

    _check_cols(df_cluster_daily, [date_col, y_col, cluster_col, cat_col])
    df = df_cluster_daily.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    if train_start:
        df = df[df[date_col] >= pd.to_datetime(train_start)]
    if train_end:
        df = df[df[date_col] <= pd.to_datetime(train_end)]

    models = {}
    for (cid, cat), g in df.groupby([cluster_col, cat_col], sort=False):
        g = g.rename(columns={date_col: "ds", y_col: "y"})[["ds","y"]].dropna()
        if len(g) < 30:
            continue  # muy corto
        m = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            changepoint_prior_scale=changepoint_prior_scale,
            holidays=holidays_df
        )
        m.fit(g)
        models[(int(cid), str(cat))] = m
    return models

def prophet_predict_cluster_windowN(
    models: Dict[Tuple[int, str], object],
    horizon_dates: pd.DatetimeIndex,
    N: int = 3
) -> pd.DataFrame:
    """
    Predice con Prophet por (cluster,cat) para fechas 'horizon_dates' y agrega a ventana N (suma de yhat en [t..t+N-1]).
    Retorna DataFrame: [ds, cluster_id, categoria_delito, yhat_N]
    """
    rows = []
    for (cid, cat), m in models.items():
        # construir futuro: cubrir desde min(horizon) hasta max(horizon)+N-1
        ds_min = horizon_dates.min()
        ds_max = horizon_dates.max() + pd.Timedelta(days=N-1)
        future = pd.DataFrame({"ds": pd.date_range(ds_min, ds_max, freq="D")})
        fcst = m.predict(future)[["ds","yhat"]]

        # rolling-sum hacia adelante: para cada t, suma t..t+N-1
        # equivalencia: yhat_N(t) = (yhat.rolling(N).sum()).shift(-(N-1))
        fcst = fcst.sort_values("ds")
        fcst["yhat_N"] = fcst["yhat"].rolling(window=N, min_periods=N).sum().shift(-(N-1))

        sub = fcst[fcst["ds"].isin(horizon_dates)][["ds","yhat_N"]].copy()
        sub["cluster_id"] = cid
        sub["categoria_delito"] = cat
        rows.append(sub)
    if not rows:
        return pd.DataFrame(columns=["ds","cluster_id","categoria_delito","yhat_N"])
    out = pd.concat(rows, ignore_index=True)
    return out




# Downscale de cluster a celdas
def make_downscale_weights_at_train_end(
    df: pd.DataFrame,
    date_col: str = "ds",
    y_col: str = "y",
    cluster_col: str = "cluster_id",
    cell_col: str = "cell_id",
    cat_col: str = "categoria_delito",
    set_col: str = "set",
    train_set_value: str = "train",
    ref_window_days: int = 28,
) -> pd.DataFrame:
    """
    Calcula pesos para downscale por (cluster,cat,cell) usando ventana histórica
    dentro del set='train', cerrada al último día de train (sin fuga).
    Retorna: [cluster_id, categoria_delito, cell_id, weight] con sum(weight) por (cluster,cat) = 1.
    """
    _check_cols(df, [date_col, y_col, cluster_col, cell_col, cat_col, set_col])
    X = df[df[set_col] == train_set_value].copy()
    X[date_col] = pd.to_datetime(X[date_col], errors="coerce")
    ds_ref = X[date_col].max()  # último día de train
    ds_ini = ds_ref - pd.Timedelta(days=ref_window_days-1)
    W = X[(X[date_col] >= ds_ini) & (X[date_col] <= ds_ref)].copy()

    grp = (W.groupby([cluster_col, cat_col, cell_col], sort=False)[y_col]
             .sum().rename("sum_ref").reset_index())
    # normalizar por (cluster,cat)
    denom = grp.groupby([cluster_col, cat_col], sort=False)["sum_ref"].transform(lambda s: s.sum() if s.sum()>0 else 1.0)
    grp["weight"] = grp["sum_ref"] / denom
    return grp[[cluster_col, cat_col, cell_col, "weight"]]



def downscale_cluster_predictions_to_cells(
    pred_clusterN: pd.DataFrame,  # [ds, cluster_id, categoria_delito, yhat_N]
    weights_df: pd.DataFrame,     # [cluster_id, categoria_delito, cell_id, weight]
) -> pd.DataFrame:
    """
    Distribuye yhat_N por cluster a celdas mediante weights.
    Retorna: [ds, cell_id, categoria_delito, yhat_N_cell]
    """
    _check_cols(pred_clusterN, ["ds","cluster_id","categoria_delito","yhat_N"])
    _check_cols(weights_df, ["cluster_id","categoria_delito","cell_id","weight"])
    out = pred_clusterN.merge(weights_df, on=["cluster_id","categoria_delito"], how="left")
    out["yhat_N_cell"] = out["yhat_N"] * out["weight"].fillna(0.0)
    return out[["ds","cell_id","categoria_delito","yhat_N_cell"]]



#  XGBoost (API nativa, agnóstica de versión) 
def fit_xgb_classifier_dmatrix(
    df_feats: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    set_col: str = "set",
    train_set_value: str = "train",
    valid_frac: float = 0.2,
    shuffle: bool = False,
    params: Optional[dict] = None,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 20,
):
    """
    Entrena clasificador XGB con DMatrix + early stopping (sin depender del wrapper sklearn).
    Retorna (booster, best_iteration, best_score)
    """
    import xgboost as xgb
    from sklearn.model_selection import train_test_split

    _check_cols(df_feats, feature_cols + [label_col, set_col])
    train_df = df_feats[df_feats[set_col] == train_set_value].dropna(subset=[label_col]).copy()
    X = train_df[feature_cols]
    y = train_df[label_col].astype(int)

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=valid_frac, shuffle=shuffle)

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_cols)
    dvalid = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)

    pos_rate = float(y_tr.mean()) if len(y_tr)>0 else 0.01
    scale_pos_weight = float((1 - pos_rate) / max(pos_rate, 1e-6))

    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.1,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "lambda": 1.0,
        "scale_pos_weight": scale_pos_weight,
    }
    if params:
        base_params.update(params)

    booster = xgb.train(
        params=base_params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )
    return booster, booster.best_iteration, booster.best_score



def predict_xgb_classifier(
    booster,
    df_feats: pd.DataFrame,
    feature_cols: List[str],
) -> pd.Series:
    """
    Predice probas con booster XGB (clasificación).
    """
    import xgboost as xgb
    _check_cols(df_feats, feature_cols)
    dtest = xgb.DMatrix(df_feats[feature_cols], feature_names=feature_cols)
    proba = booster.predict(dtest, iteration_range=(0, getattr(booster, "best_iteration", 0)+1))
    return pd.Series(proba, index=df_feats.index)



#  SARIMAX (opcional, por cluster) 
def fit_sarimax_by_cluster_category(
    df_cluster_daily: pd.DataFrame,
    date_col: str = "ds",
    y_col: str = "y_cluster",
    cluster_col: str = "cluster_id",
    cat_col: str = "categoria_delito",
    order=(1,0,1),
    seasonal_order=(1,0,1,7),
    enforce_stationarity=False,
    enforce_invertibility=False,
) -> Dict[Tuple[int,str], object]:
    """
    Entrena SARIMAX por (cluster, categoría). Devuelve dict de modelos ajustados.
    Nota: costoso; usa en clusters con serie suficiente.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    _check_cols(df_cluster_daily, [date_col, y_col, cluster_col, cat_col])
    df = df_cluster_daily.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    models = {}
    for (cid, cat), g in df.groupby([cluster_col, cat_col], sort=False):
        g = g.sort_values(date_col)
        if len(g) < 60:
            continue
        endog = g[y_col].astype(float).values
        try:
            mod = SARIMAX(endog,
                          order=order,
                          seasonal_order=seasonal_order,
                          enforce_stationarity=enforce_stationarity,
                          enforce_invertibility=enforce_invertibility)
            res = mod.fit(disp=False)
            models[(int(cid), str(cat))] = (res, g[date_col].iloc[0])  # guardo fecha inicio para indexar
        except Exception:
            continue
    return models



def sarimax_predict_cluster_windowN(
    models: Dict[Tuple[int,str], object],
    horizon_dates: pd.DatetimeIndex,
    N: int = 3,
) -> pd.DataFrame:
    """
    Predice ventana N con modelos SARIMAX por (cluster,cat) para horizon_dates.
    Necesita mapear fecha índice relativo.
    """
    rows = []
    for (cid, cat), payload in models.items():
        res, start_date = payload
        # necesitamos pronosticar hasta max(horizon)+N-1
        ds_min = horizon_dates.min()
        ds_max = horizon_dates.max() + pd.Timedelta(days=N-1)
        steps = (ds_max - start_date).days + 1
        if steps <= 0:
            continue
        fcst = res.get_forecast(steps=steps)
        yhat = fcst.predicted_mean
        idx = pd.date_range(start_date, periods=len(yhat), freq="D")
        dfp = pd.DataFrame({"ds": idx, "yhat": np.asarray(yhat, dtype=float)})
        dfp = dfp[dfp["ds"] >= ds_min].copy()
        dfp = dfp.sort_values("ds")
        dfp["yhat_N"] = dfp["yhat"].rolling(window=N, min_periods=N).sum().shift(-(N-1))
        sub = dfp[dfp["ds"].isin(horizon_dates)][["ds","yhat_N"]].copy()
        sub["cluster_id"] = cid
        sub["categoria_delito"] = cat
        rows.append(sub)
    if not rows:
        return pd.DataFrame(columns=["ds","cluster_id","categoria_delito","yhat_N"])
    return pd.concat(rows, ignore_index=True)




# Ranking Top-k por categoría (cell-level) 
def rank_hotspots_topk_from_scores(
    scores_df: pd.DataFrame,   # [cell_id, categoria_delito, score] (+ opcional ds)
    k: int = 5,
    method: str = "h3",
    date_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Construye top-k por categoría (y por fecha si viene 'date_col').
    Retorna centroides lat/lon listos para mapa y evaluación Hit@k.
    """
    _check_cols(scores_df, ["cell_id","categoria_delito","score"])
    if date_col and date_col in scores_df.columns:
        frames = []
        for ds, g in scores_df.groupby(date_col, sort=False):
            frames.append(
                _rank_hotspots_single(g, k=k, method=method, date_val=ds, date_col=date_col)
            )
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        return _rank_hotspots_single(scores_df, k=k, method=method, date_val=None, date_col=None)



def _rank_hotspots_single(g: pd.DataFrame, k: int, method: str, date_val, date_col):
    g2 = g.sort_values("score", ascending=False).groupby("categoria_delito", sort=False).head(k).copy()
    # centroids
    cent = cells_to_centroids(g2["cell_id"].tolist(), method=method)
    g2 = g2.merge(cent, on="cell_id", how="left")
    if date_col and date_val is not None:
        g2[date_col] = date_val
    # Normalizar nombres
    outcols = ["cell_id","categoria_delito","score","lon","lat"]
    if date_col and date_val is not None:
        outcols.append(date_col)
    return g2[outcols]





def eval_classification_temporal_series(
    scores_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    name: str = "model",
    date_col: str = "ds",
    cat_col: str = "categoria_delito",
    label_col: str = "y_bin",
):
    """
    Evalúa un modelo de clasificación temporal comparando scores y labels.
    Devuelve métricas globales y por categoría.
    """
    # Merge
    df_eval = labels_df.merge(
        scores_df.rename(columns={"score": f"score_{name}"}),
        on=[date_col, "cell_id", cat_col],
        how="inner"
    )

    y_true = df_eval[label_col].astype(int).values
    y_score = df_eval[f"score_{name}"].astype(float).values

    out = {}
    # Global metrics
    try:
        out["pr_auc"] = average_precision_score(y_true, y_score)
    except Exception:
        out["pr_auc"] = np.nan
    try:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
    except Exception:
        out["roc_auc"] = np.nan
    try:
        out["brier"] = brier_score_loss(y_true, y_score)
    except Exception:
        out["brier"] = np.nan

    # Por categoría
    by_cat = []
    for cat, g in df_eval.groupby(cat_col):
        yt = g[label_col].values
        ys = g[f"score_{name}"].values
        if len(np.unique(yt)) < 2:
            by_cat.append({
                cat_col: cat,
                "pr_auc": np.nan,
                "roc_auc": np.nan,
                "brier": np.nan,
                "n": len(g)
            })
            continue
        by_cat.append({
            cat_col: cat,
            "pr_auc": average_precision_score(yt, ys),
            "roc_auc": roc_auc_score(yt, ys),
            "brier": brier_score_loss(yt, ys),
            "n": len(g)
        })

    by_cat_df = pd.DataFrame(by_cat).sort_values("pr_auc", ascending=False)
    return out, by_cat_df, df_eval




def eval_hitk_over_days(
    topk_all: pd.DataFrame,
    df_cells_points: pd.DataFrame,
    dates_index: pd.Series,
    radius_m: float,
    date_col: str = "ds",
    cat_col: str = "categoria_delito"
) -> pd.DataFrame:
    """
    Evalúa Hit@k@R (radio en metros) para un conjunto de predicciones top-k diarias.
    
    Parámetros
    ----------
    topk_all : pd.DataFrame
        Predicciones top-k con columnas [date_col, cell_id, lat, lon, categoria_delito].
    df_cells_points : pd.DataFrame
        Incidentes reales con columnas [fecha_evento, latitud, longitud, categoria_delito].
    dates_index : iterable
        Fechas a evaluar.
    radius_m : float
        Radio de búsqueda en metros.
    date_col : str, default="ds"
        Columna de fecha en el dataframe de predicciones.
    cat_col : str, default="categoria_delito"
        Columna de categoría del delito.
    
    Retorna
    -------
    pd.DataFrame con columnas:
        [categoria_delito, n_events, n_hits, hit_rate, date_col]
    """
    rows = []
    for d in dates_index:
        # Filtrar predicciones y observaciones para la fecha d
        pred_d = topk_all[topk_all[date_col] == d]
        truth_d = df_cells_points[df_cells_points["fecha_evento"] == d][["latitud", "longitud", cat_col]]
        
        # Saltar si alguna está vacía
        if pred_d.empty or truth_d.empty:
            continue
        
        # Calcular Hit@k@R
        res = hit_at_k_radius(
            topk_df=pred_d,
            truth_df=truth_d,
            cat_col=cat_col,
            radius_m=radius_m
        )
        res[date_col] = d
        rows.append(res)
    
    # Concatenar resultados o devolver DataFrame vacío
    if not rows:
        return pd.DataFrame(columns=[cat_col, "n_events", "n_hits", "hit_rate", date_col])
    return pd.concat(rows, ignore_index=True)





def summarise_hitk(df_hit: pd.DataFrame, date_col: str = "ds") -> dict:
    """
    Resume Hit@k@R agregando por día.
    Espera columnas: [categoria_delito, n_events, n_hits, hit_rate, date_col]
    """
    if df_hit is None or df_hit.empty:
        return {"days_eval": 0, "hit_rate_mean": np.nan, "hit_rate_median": np.nan}
    if date_col not in df_hit.columns:
        raise KeyError(f"'{date_col}' no está en df_hit. Pasa el nombre correcto vía date_col=")
    return {
        "days_eval": int(df_hit[date_col].nunique()),
        "hit_rate_mean": float(df_hit["hit_rate"].mean()),
        "hit_rate_median": float(df_hit["hit_rate"].median()),
    }



def smape(y, yhat):
    denom = (np.abs(y) + np.abs(yhat))
    denom[denom==0] = 1.0
    return np.mean(2.0 * np.abs(yhat - y) / denom)







def eval_regression_counts(df_true_pred, y_true_col, y_pred_col, model_name="model"):
    """
    Evalúa métricas de regresión para conteo de eventos.
    Retorna dict con RMSE, MAE, R2, sMAPE, bias y correlación temporal.
    """
    y_true = df_true_pred[y_true_col].astype(float).values
    y_pred = df_true_pred[y_pred_col].astype(float).values

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # sMAPE (tu versión o genérica)
    smape_val = smape(y_true, y_pred)

    # Bias (promedio de error relativo)
    bias = np.mean(y_pred - y_true)

    # Correlación temporal (si hay fechas)
    corr = np.corrcoef(y_true, y_pred)[0,1]

    metrics = {
        "model": model_name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "smape": smape_val,
        "bias": bias,
        "corr": corr
    }

    print(f"{model_name} — RMSE: {rmse:.3f} | MAE: {mae:.3f} | R²: {r2:.3f} | "
          f"sMAPE: {smape_val:.3f} | Corr: {corr:.3f} | Bias: {bias:.3f}")

    return metrics







def eval_regression_counts_by_category(df_true_pred, y_true_col, y_pred_col, cat_col, model_name="model"):
    """
    Evalúa métricas de regresión para conteo de eventos por categoría.
    Retorna DataFrame con RMSE, MAE, R², sMAPE, bias y correlación.
    """
    rows = []
    cats = df_true_pred[cat_col].unique()
    
    for cat in cats:
        dfc = df_true_pred[df_true_pred[cat_col] == cat].dropna(subset=[y_true_col, y_pred_col])
        if len(dfc) < 5:
            continue

        y_true = dfc[y_true_col].astype(float).values
        y_pred = dfc[y_pred_col].astype(float).values

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        smape_val = smape(y_true, y_pred)
        bias = np.mean(y_pred - y_true)
        corr = np.corrcoef(y_true, y_pred)[0,1] if len(y_true) > 1 else np.nan

        rows.append({
            "model": model_name,
            cat_col: cat,
            "n": len(dfc),
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "smape": smape_val,
            "bias": bias,
            "corr": corr
        })

    df_metrics = pd.DataFrame(rows)
    df_metrics = df_metrics.sort_values("r2", ascending=False).reset_index(drop=True)
    
    print(f"Desempeño de {model_name}")
    display(df_metrics.round(4))
    
    return df_metrics
