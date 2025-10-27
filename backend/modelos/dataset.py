import pandas as pd
import numpy as np
from typing import Dict, List, Any

class GestorDataset:

    def __init__(self, ruta_archivo: str):
        self.df = pd.read_csv(ruta_archivo)
        self.df_original = self.df.copy()

    def obtener_informacion(self) -> Dict[str, Any]:
        return {
            'filas': len(self.df),
            'columnas': len(self.df.columns),
            'nombres_columnas': self.df.columns.tolist()
        }

    def obtener_estadisticas_columnas(self) -> List[Dict[str, Any]]:
        estadisticas = []
        for columna in self.df.columns:
            stat = {
                'nombre': columna,
                'tipo': str(self.df[columna].dtype),
                'valores_nulos': int(self.df[columna].isnull().sum()),
                'valores_unicos': int(self.df[columna].nunique())
            }
            if pd.api.types.is_numeric_dtype(self.df[columna]):
                stat['promedio'] = float(self.df[columna].mean())
                stat['min'] = float(self.df[columna].min())
                stat['max'] = float(self.df[columna].max())
                stat['desviacion'] = float(self.df[columna].std())
            estadisticas.append(stat)
        return estadisticas

    def obtener_vista_previa(self, n_filas: int = 20) -> List[Dict]:
        preview_df = self.df.head(n_filas).copy()
        preview_dict = preview_df.to_dict('records')
        
        result = []
        for record in preview_dict:
            new_record = {}
            for key, value in record.items():
                if pd.isna(value):
                    new_record[key] = '[NULL]'
                elif isinstance(value, float):
                    new_record[key] = round(value, 4)
                else:
                    new_record[key] = str(value)
            result.append(new_record)
        
        return result

    def obtener_estadisticas_datos(self) -> Dict[str, Any]:
        return {
            'total_filas': len(self.df),
            'total_columnas': len(self.df.columns),
            'total_nulos': int(self.df.isnull().sum().sum()),
            'total_duplicados': int(self.df.duplicated().sum()),
            'porcentaje_nulos': round((self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns)) * 100), 2)
        }

    def guardar(self, ruta: str):
        self.df.to_csv(ruta, index=False)