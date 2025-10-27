import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Any, Tuple

class ServicioLimpieza:

    @staticmethod
    def eliminar_valores_nulos(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        filas_antes = len(df)
        df_limpio = df.dropna()
        filas_eliminadas = filas_antes - len(df_limpio)
        return df_limpio, filas_eliminadas

    @staticmethod
    def eliminar_duplicados(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        duplicados = df.duplicated().sum()
        df_limpio = df.drop_duplicates()
        return df_limpio, int(duplicados)

    @staticmethod
    def normalizar_datos(df: pd.DataFrame) -> pd.DataFrame:
        scaler = StandardScaler()
        columnas_numericas = df.select_dtypes(include=[np.number]).columns
        if len(columnas_numericas) > 0:
            df[columnas_numericas] = scaler.fit_transform(df[columnas_numericas])
        return df

    @staticmethod
    def codificar_categoricas(df: pd.DataFrame) -> pd.DataFrame:
        columnas_categoricas = df.select_dtypes(include=['object']).columns
        label_encoders = {}
        for columna in columnas_categoricas:
            le = LabelEncoder()
            df[columna] = le.fit_transform(df[columna].astype(str))
            label_encoders[columna] = le
        return df

    @staticmethod
    def detectar_outliers(df: pd.DataFrame, umbral: float = 3.0) -> Tuple[pd.DataFrame, int]:
        filas_antes = len(df)
        columnas_numericas = df.select_dtypes(include=[np.number]).columns
        for columna in columnas_numericas:
            if df[columna].std() != 0:
                z_scores = np.abs((df[columna] - df[columna].mean()) / df[columna].std())
                df = df[z_scores < umbral]
        filas_eliminadas = filas_antes - len(df)
        return df, filas_eliminadas

    @staticmethod
    def obtener_estadisticas_limpieza(df_original: pd.DataFrame, df_limpio: pd.DataFrame,
                                      nulos_eliminados: int, duplicados_eliminados: int,
                                      outliers_eliminados: int) -> Dict[str, Any]:
        filas_originales = len(df_original)
        filas_limpias = len(df_limpio)
        filas_eliminadas = filas_originales - filas_limpias

        nulos_por_columna = {}
        for col in df_original.columns:
            nulos = int(df_original[col].isnull().sum())
            if nulos > 0:
                nulos_por_columna[col] = nulos

        columnas_eliminadas = list(set(df_original.columns) - set(df_limpio.columns))

        return {
            'filas_originales': filas_originales,
            'filas_limpias': filas_limpias,
            'filas_eliminadas': filas_eliminadas,
            'porcentaje_datos_eliminados': round((filas_eliminadas / filas_originales * 100), 2) if filas_originales > 0 else 0,
            'nulos_por_columna': nulos_por_columna,
            'total_nulos': sum(nulos_por_columna.values()),
            'duplicados_detectados': duplicados_eliminados,
            'columnas_eliminadas': columnas_eliminadas,
            'nulos_eliminados': nulos_eliminados,
            'duplicados_eliminados': duplicados_eliminados,
            'outliers_eliminados': outliers_eliminados
        }

    @staticmethod
    def aplicar_limpieza(df: pd.DataFrame, operaciones: Dict[str, bool]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        df_original = df.copy()
        df_limpio = df.copy()

        nulos_eliminados = 0
        duplicados_eliminados = 0
        outliers_eliminados = 0

        if operaciones.get('eliminar_nulos', False):
            df_limpio, nulos_eliminados = ServicioLimpieza.eliminar_valores_nulos(df_limpio)

        if operaciones.get('eliminar_duplicados', False):
            df_limpio, duplicados_eliminados = ServicioLimpieza.eliminar_duplicados(df_limpio)

        if operaciones.get('detectar_outliers', False):
            df_limpio, outliers_eliminados = ServicioLimpieza.detectar_outliers(df_limpio)

        if operaciones.get('codificar_categoricas', False):
            df_limpio = ServicioLimpieza.codificar_categoricas(df_limpio)

        if operaciones.get('normalizar', False):
            df_limpio = ServicioLimpieza.normalizar_datos(df_limpio)

        estadisticas = ServicioLimpieza.obtener_estadisticas_limpieza(
            df_original, df_limpio, nulos_eliminados, duplicados_eliminados, outliers_eliminados
        )

        return df_limpio, estadisticas