import numpy as np
import pandas as pd
from typing import List, Dict, Any

class ServicioVisualizacion:
   
    @staticmethod
    def generar_datos_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        return {
            'matriz': cm.tolist(),
            'etiquetas': list(range(len(cm)))
        }
   
    @staticmethod
    def generar_curva_roc(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, umbrales = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        return {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': float(roc_auc)
        }
   
    @staticmethod
    def generar_distribucion_errores(y_true: np.ndarray, y_pred: np.ndarray) -> List[float]:
        errores = y_true - y_pred
        return errores.tolist()
   
    @staticmethod
    def generar_importancia_features(modelo: Any, nombres_features: List[str]) -> List[Dict[str, Any]]:
        if hasattr(modelo, 'feature_importances_'):
            importancias = modelo.feature_importances_
            return [
                {'feature': nombre, 'importancia': float(imp)}
                for nombre, imp in zip(nombres_features, importancias)
            ]
        return []
   
    @staticmethod
    def generar_mapa_calor_precios(df: pd.DataFrame) -> List[Dict[str, Any]]:
        if 'zona' in df.columns and 'precio' in df.columns:
            mapa_calor = df.groupby('zona')['precio'].mean().reset_index()
            mapa_calor.columns = ['zona', 'precio_promedio']
            mapa_calor['zona'] = mapa_calor['zona'].apply(lambda x: f'Zona {int(x)}')
            return mapa_calor.to_dict('records')
        return []
   
    @staticmethod
    def generar_datos_3d(df: pd.DataFrame, max_puntos: int = 100) -> List[Dict[str, Any]]:
        if 'area_m2' in df.columns and 'precio' in df.columns and 'habitaciones' in df.columns:
            df_sample = df[['area_m2', 'precio', 'habitaciones']].dropna().sample(min(max_puntos, len(df)))
            datos_3d = df_sample.to_dict('records')
            for dato in datos_3d:
                dato['area_m2'] = round(float(dato['area_m2']), 2)
                dato['precio'] = round(float(dato['precio']), 2)
                dato['habitaciones'] = int(dato['habitaciones'])
            return datos_3d
        return []
   
    @staticmethod
    def generar_histograma_precio(df: pd.DataFrame, bins: int = 10) -> List[Dict[str, Any]]:
        if 'precio' in df.columns:
            precios = df['precio'].dropna()
            min_precio = precios.min()
            max_precio = precios.max()
            step = (max_precio - min_precio) / bins
           
            histogram = []
            for i in range(bins):
                inicio = min_precio + i * step
                fin = inicio + step
                count = len(precios[(precios >= inicio) & (precios < fin)])
                histogram.append({
                    'rango': f'{int(inicio/1000)}K',
                    'frecuencia': int(count)
                })
           
            return histogram
        return []
   
    @staticmethod
    def generar_boxplot_zonas(df: pd.DataFrame) -> List[Dict[str, Any]]:
        if 'zona' in df.columns and 'precio' in df.columns:
            boxplot_data = []
            for zona in sorted(df['zona'].unique()):
                precios_zona = df[df['zona'] == zona]['precio'].dropna()
                if len(precios_zona) > 0:
                    q1 = precios_zona.quantile(0.25)
                    mediana = precios_zona.median()
                    q3 = precios_zona.quantile(0.75)
                    min_val = precios_zona.min()
                    max_val = precios_zona.max()
                    iqr = q3 - q1
                   
                    boxplot_data.append({
                        'zona': f'Zona {int(zona)}',
                        'min': round(float(min_val), 2),
                        'q1': round(float(q1), 2),
                        'mediana': round(float(mediana), 2),
                        'q3': round(float(q3), 2),
                        'max': round(float(max_val), 2),
                        'rango': round(float(max_val - min_val), 2),
                        'iqr': round(float(iqr), 2)
                    })
           
            return boxplot_data
        return []
   
    @staticmethod
    def generar_serie_temporal(df: pd.DataFrame) -> List[Dict[str, Any]]:
        if 'ano_construccion' in df.columns and 'precio' in df.columns:
            serie_temporal = df.groupby('ano_construccion')['precio'].mean().reset_index()
            serie_temporal.columns = ['ano', 'precio_promedio']
            serie_temporal = serie_temporal.sort_values('ano')
            serie_temporal['precio_promedio'] = serie_temporal['precio_promedio'].round(2)
            return serie_temporal.to_dict('records')
        return []