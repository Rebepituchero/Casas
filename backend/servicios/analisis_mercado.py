import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any, Tuple
import logging

logging.getLogger().setLevel(logging.ERROR)

class ServicioAnalisisMercado:
    
    @staticmethod
    def segmentar_mercado(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if 'precio' not in df.columns:
            return df, {}
        
        df_clean = df.copy()
        df_clean = df_clean[df_clean['precio'].notna()]
        
        if len(df_clean) == 0:
            return df, {}
        
        q33 = df_clean['precio'].quantile(0.33)
        q66 = df_clean['precio'].quantile(0.66)
        
        def clasificar_segmento(precio):
            if precio <= q33:
                return 'economico'
            elif precio <= q66:
                return 'medio'
            else:
                return 'lujo'
        
        df_clean['segmento_mercado'] = df_clean['precio'].apply(clasificar_segmento)
        
        segmentos_stats = {}
        for segmento in ['economico', 'medio', 'lujo']:
            seg_data = df_clean[df_clean['segmento_mercado'] == segmento]
            if len(seg_data) > 0:
                segmentos_stats[segmento] = {
                    'cantidad': int(len(seg_data)),
                    'precio_promedio': round(float(seg_data['precio'].mean()), 2),
                    'precio_min': round(float(seg_data['precio'].min()), 2),
                    'precio_max': round(float(seg_data['precio'].max()), 2),
                    'porcentaje': round(len(seg_data) / len(df_clean) * 100, 2)
                }
        
        if 'area_m2' in df_clean.columns:
            for segmento in segmentos_stats:
                seg_data = df_clean[df_clean['segmento_mercado'] == segmento]
                if len(seg_data) > 0 and seg_data['area_m2'].notna().any():
                    segmentos_stats[segmento]['area_promedio'] = round(float(seg_data['area_m2'].mean()), 2)
        
        return df_clean, segmentos_stats
    
    @staticmethod
    def detectar_anomalias_precios(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        columnas_usar = ['precio']
        columnas_opcionales = ['area_m2', 'habitaciones', 'banos']
        
        if 'precio' not in df.columns:
            return [], {'gangas': 0, 'sobrevaloradas': 0, 'normales': 0}
        
        for col in columnas_opcionales:
            if col in df.columns:
                columnas_usar.append(col)
        
        df_clean = df[columnas_usar].dropna()
        
        if len(df_clean) < 10:
            return [], {'gangas': 0, 'sobrevaloradas': 0, 'normales': 0}
        
        precio_medio = df_clean['precio'].median()
        precio_std = df_clean['precio'].std()
        
        limite_inferior = precio_medio - 2 * precio_std
        limite_superior = precio_medio + 2 * precio_std
        
        anomalias_detectadas = []
        conteo = {'gangas': 0, 'sobrevaloradas': 0, 'normales': 0}
        
        for idx, row in df_clean.iterrows():
            if row['precio'] < limite_inferior:
                tipo = 'ganga'
                conteo['gangas'] += 1
                
                anomalia_info = {
                    'indice': int(idx),
                    'precio': round(float(row['precio']), 2),
                    'tipo': tipo,
                    'score': round(float((precio_medio - row['precio']) / precio_std), 4),
                    'desviacion_media': round(float((precio_medio - row['precio']) / precio_medio * 100), 2)
                }
                
                for col in columnas_usar:
                    if col != 'precio':
                        anomalia_info[col] = round(float(row[col]), 2)
                
                anomalias_detectadas.append(anomalia_info)
                
            elif row['precio'] > limite_superior:
                tipo = 'sobrevalorada'
                conteo['sobrevaloradas'] += 1
                
                anomalia_info = {
                    'indice': int(idx),
                    'precio': round(float(row['precio']), 2),
                    'tipo': tipo,
                    'score': round(float((row['precio'] - precio_medio) / precio_std), 4),
                    'desviacion_media': round(float((row['precio'] - precio_medio) / precio_medio * 100), 2)
                }
                
                for col in columnas_usar:
                    if col != 'precio':
                        anomalia_info[col] = round(float(row[col]), 2)
                
                anomalias_detectadas.append(anomalia_info)
            else:
                conteo['normales'] += 1
        
        anomalias_detectadas.sort(key=lambda x: abs(x['desviacion_media']), reverse=True)
        
        return anomalias_detectadas[:50], conteo
    
    @staticmethod
    def calcular_score_inversion(df: pd.DataFrame) -> List[Dict[str, Any]]:
        if 'precio' not in df.columns:
            return []
        
        try:
            df_clean = df[df['precio'].notna()].copy()
            
            if len(df_clean) == 0:
                return []
            
            df_clean['score_inversion'] = 50.0
            
            if 'zona' in df_clean.columns:
                try:
                    df_clean['zona'] = pd.to_numeric(df_clean['zona'], errors='coerce')
                    zona_valida = df_clean['zona'].notna()
                    if zona_valida.sum() > 0:
                        zona_promedio = df_clean[zona_valida].groupby('zona')['precio'].transform('mean')
                        precio_median = df_clean['precio'].median()
                        df_clean.loc[zona_valida, 'zona_premium'] = (zona_promedio > precio_median).astype(float)
                        df_clean['score_inversion'] = df_clean['score_inversion'] + (df_clean.get('zona_premium', pd.Series(0.0, index=df_clean.index)).fillna(0) * 15)
                except Exception:
                    pass
            
            if 'ano_construccion' in df_clean.columns:
                try:
                    df_clean['ano_construccion'] = pd.to_numeric(df_clean['ano_construccion'], errors='coerce')
                    ano_valido = df_clean['ano_construccion'].notna()
                    if ano_valido.sum() > 0:
                        ano_actual = 2025
                        df_clean.loc[ano_valido, 'antiguedad'] = ano_actual - df_clean.loc[ano_valido, 'ano_construccion']
                        df_clean.loc[ano_valido, 'score_antiguedad'] = df_clean.loc[ano_valido, 'antiguedad'].apply(
                            lambda x: 20.0 if x < 5 else (15.0 if x < 10 else (10.0 if x < 20 else 5.0))
                        )
                        df_clean['score_inversion'] = df_clean['score_inversion'] + df_clean.get('score_antiguedad', pd.Series(0.0, index=df_clean.index)).fillna(0)
                except Exception:
                    pass
            
            if 'estado_conservacion' in df_clean.columns:
                try:
                    df_clean['estado_conservacion'] = pd.to_numeric(df_clean['estado_conservacion'], errors='coerce')
                    estado_valido = df_clean['estado_conservacion'].notna()
                    if estado_valido.sum() > 0:
                        df_clean.loc[estado_valido, 'score_conservacion'] = df_clean.loc[estado_valido, 'estado_conservacion'] * 5.0
                        df_clean['score_inversion'] = df_clean['score_inversion'] + df_clean.get('score_conservacion', pd.Series(0.0, index=df_clean.index)).fillna(0)
                except Exception:
                    pass
            
            if 'area_m2' in df_clean.columns:
                try:
                    df_clean['area_m2'] = pd.to_numeric(df_clean['area_m2'], errors='coerce')
                    area_valida = (df_clean['area_m2'].notna()) & (df_clean['area_m2'] > 0)
                    if area_valida.sum() > 0 and df_clean.loc[area_valida, 'area_m2'].std() > 0:
                        precio_m2 = df_clean.loc[area_valida, 'precio'] / df_clean.loc[area_valida, 'area_m2']
                        percentil = precio_m2.rank(pct=True)
                        df_clean.loc[area_valida, 'score_precio'] = (1 - percentil) * 20.0
                        df_clean['score_inversion'] = df_clean['score_inversion'] + df_clean.get('score_precio', pd.Series(0.0, index=df_clean.index)).fillna(0)
                except Exception:
                    pass
            
            amenidades = ['garaje', 'ascensor', 'balcon', 'piscina', 'jardin']
            for amenidad in amenidades:
                if amenidad in df_clean.columns:
                    try:
                        df_clean[amenidad] = pd.to_numeric(df_clean[amenidad], errors='coerce').fillna(0)
                        df_clean['score_inversion'] = df_clean['score_inversion'] + (df_clean[amenidad] * 2.0)
                    except Exception:
                        pass
            
            if 'distancia_centro_km' in df_clean.columns:
                try:
                    df_clean['distancia_centro_km'] = pd.to_numeric(df_clean['distancia_centro_km'], errors='coerce')
                    dist_valida = df_clean['distancia_centro_km'].notna()
                    if dist_valida.sum() > 0:
                        df_clean.loc[dist_valida, 'score_ubicacion'] = df_clean.loc[dist_valida, 'distancia_centro_km'].apply(
                            lambda x: 10.0 if x < 3 else (7.0 if x < 7 else (4.0 if x < 15 else 2.0))
                        )
                        df_clean['score_inversion'] = df_clean['score_inversion'] + df_clean.get('score_ubicacion', pd.Series(0.0, index=df_clean.index)).fillna(0)
                except Exception:
                    pass
            
            df_clean['score_inversion'] = df_clean['score_inversion'].clip(0, 100)
            
            def clasificar_potencial(score):
                if score >= 80:
                    return 'Excelente'
                elif score >= 65:
                    return 'Muy Bueno'
                elif score >= 50:
                    return 'Bueno'
                elif score >= 35:
                    return 'Regular'
                else:
                    return 'Bajo'
            
            df_clean['potencial_revalorizacion'] = df_clean['score_inversion'].apply(clasificar_potencial)
            
            resultados = []
            for idx, row in df_clean.iterrows():
                try:
                    resultado = {
                        'indice': int(idx),
                        'precio': round(float(row['precio']), 2),
                        'score_inversion': round(float(row['score_inversion']), 2),
                        'potencial': row['potencial_revalorizacion']
                    }
                    
                    if 'area_m2' in row.index and pd.notna(row.get('area_m2')):
                        resultado['area_m2'] = round(float(row['area_m2']), 2)
                    if 'zona' in row.index and pd.notna(row.get('zona')):
                        resultado['zona'] = int(float(row['zona']))
                    if 'ano_construccion' in row.index and pd.notna(row.get('ano_construccion')):
                        resultado['ano_construccion'] = int(float(row['ano_construccion']))
                    
                    resultados.append(resultado)
                except Exception:
                    continue
            
            resultados.sort(key=lambda x: x['score_inversion'], reverse=True)
            
            return resultados
            
        except Exception:
            return []