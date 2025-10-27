from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import requests
from io import StringIO
from supabase import create_client
from config import Config
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

analisis_bp = Blueprint('analisis', __name__)
supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)

@analisis_bp.route('/datasets/<dataset_id>/analisis-radar', methods=['GET'])
def obtener_analisis_radar(dataset_id):
    try:
        dataset = supabase.table('datasets').select('*').eq('id', dataset_id).execute()
        if not dataset.data:
            return jsonify([])
        
        respuesta_archivo = requests.get(dataset.data[0]['archivo_url'], timeout=30)
        respuesta_archivo.raise_for_status()
        df = pd.read_csv(StringIO(respuesta_archivo.text))
        
        columnas_numericas = ['area_m2', 'habitaciones', 'banos', 'precio', 'ano_construccion']
        columnas_disponibles = [col for col in columnas_numericas if col in df.columns]
        
        if len(columnas_disponibles) < 3:
            return jsonify([])
        
        df_clean = df[columnas_disponibles].dropna()
        if len(df_clean) == 0:
            return jsonify([])
        
        df_sample = df_clean.head(5)
        
        scaler = StandardScaler()
        df_normalized = pd.DataFrame(
            scaler.fit_transform(df_sample),
            columns=df_sample.columns,
            index=df_sample.index
        )
        
        df_normalized = ((df_normalized - df_normalized.min()) / (df_normalized.max() - df_normalized.min()) * 100)
        
        datos_radar = []
        for idx, row in df_normalized.iterrows():
            casa_datos = {
                'casa': f'Casa {idx + 1}',
                'metricas': []
            }
            for col in df_normalized.columns:
                casa_datos['metricas'].append({
                    'metrica': col.replace('_', ' ').title(),
                    'valor': round(float(row[col]), 2)
                })
            datos_radar.append(casa_datos)
        
        return jsonify(datos_radar)
    except Exception as e:
        print(f"Error en analisis-radar: {str(e)}")
        return jsonify([])

@analisis_bp.route('/datasets/<dataset_id>/sankey-flujo', methods=['GET'])
def obtener_sankey_flujo(dataset_id):
    try:
        dataset = supabase.table('datasets').select('*').eq('id', dataset_id).execute()
        if not dataset.data:
            return jsonify({'nodes': [], 'links': []})
        
        respuesta_archivo = requests.get(dataset.data[0]['archivo_url'], timeout=30)
        respuesta_archivo.raise_for_status()
        df = pd.read_csv(StringIO(respuesta_archivo.text))
        
        columnas_requeridas = ['area_m2', 'habitaciones', 'precio']
        columnas_disponibles = [col for col in columnas_requeridas if col in df.columns]
        
        if len(columnas_disponibles) < 3:
            return jsonify({'nodes': [], 'links': []})
        
        zona_col = None
        if 'zona' in df.columns:
            zona_col = 'zona'
        
        if zona_col:
            columnas_disponibles.append(zona_col)
        
        df_clean = df[columnas_disponibles].dropna()
        if len(df_clean) == 0:
            return jsonify({'nodes': [], 'links': []})
        
        if zona_col and df_clean[zona_col].dtype == 'object':
            le = LabelEncoder()
            df_clean[zona_col] = le.fit_transform(df_clean[zona_col].astype(str))
        
        df_clean['categoria_area'] = pd.cut(df_clean['area_m2'], bins=3, labels=['Pequeño', 'Mediano', 'Grande'])
        df_clean['categoria_habitaciones'] = pd.cut(df_clean['habitaciones'], bins=3, labels=['1-2 Hab', '3-4 Hab', '5+ Hab'])
        df_clean['categoria_precio'] = pd.cut(df_clean['precio'], bins=3, labels=['Económico', 'Medio', 'Premium'])
        
        nodes = []
        node_map = {}
        node_id = 0
        
        for categoria in df_clean['categoria_area'].unique():
            if pd.notna(categoria):
                nodes.append({'id': node_id, 'name': f'Área: {categoria}'})
                node_map[f'area_{categoria}'] = node_id
                node_id += 1
        
        for categoria in df_clean['categoria_habitaciones'].unique():
            if pd.notna(categoria):
                nodes.append({'id': node_id, 'name': f'Hab: {categoria}'})
                node_map[f'hab_{categoria}'] = node_id
                node_id += 1
        
        for categoria in df_clean['categoria_precio'].unique():
            if pd.notna(categoria):
                nodes.append({'id': node_id, 'name': f'Precio: {categoria}'})
                node_map[f'precio_{categoria}'] = node_id
                node_id += 1
        
        links = []
        
        flujo_area_hab = df_clean.groupby(['categoria_area', 'categoria_habitaciones'], observed=True).size().reset_index(name='value')
        for _, row in flujo_area_hab.iterrows():
            if pd.notna(row['categoria_area']) and pd.notna(row['categoria_habitaciones']):
                links.append({
                    'source': node_map[f'area_{row["categoria_area"]}'],
                    'target': node_map[f'hab_{row["categoria_habitaciones"]}'],
                    'value': int(row['value'])
                })
        
        flujo_hab_precio = df_clean.groupby(['categoria_habitaciones', 'categoria_precio'], observed=True).size().reset_index(name='value')
        for _, row in flujo_hab_precio.iterrows():
            if pd.notna(row['categoria_habitaciones']) and pd.notna(row['categoria_precio']):
                links.append({
                    'source': node_map[f'hab_{row["categoria_habitaciones"]}'],
                    'target': node_map[f'precio_{row["categoria_precio"]}'],
                    'value': int(row['value'])
                })
        
        return jsonify({'nodes': nodes, 'links': links})
    except Exception as e:
        print(f"Error en sankey-flujo: {str(e)}")
        return jsonify({'nodes': [], 'links': []})

@analisis_bp.route('/datasets/<dataset_id>/grafico-burbujas', methods=['GET'])
def obtener_grafico_burbujas(dataset_id):
    try:
        dataset = supabase.table('datasets').select('*').eq('id', dataset_id).execute()
        if not dataset.data:
            return jsonify([])
        
        respuesta_archivo = requests.get(dataset.data[0]['archivo_url'], timeout=30)
        respuesta_archivo.raise_for_status()
        df = pd.read_csv(StringIO(respuesta_archivo.text))
        
        columnas_requeridas = ['area_m2', 'precio']
        if not all(col in df.columns for col in columnas_requeridas):
            return jsonify([])
        
        zona_col = None
        if 'zona' in df.columns:
            zona_col = 'zona'
        elif 'region' in df.columns:
            zona_col = 'region'
        elif 'distrito' in df.columns:
            zona_col = 'distrito'
        
        columnas_usar = columnas_requeridas.copy()
        if zona_col:
            columnas_usar.append(zona_col)
        
        df_clean = df[columnas_usar].dropna()
        if len(df_clean) == 0:
            return jsonify([])
        
        if zona_col:
            if df_clean[zona_col].dtype == 'object':
                le = LabelEncoder()
                df_clean['zona_numerica'] = le.fit_transform(df_clean[zona_col].astype(str))
                df_clean['zona_label'] = df_clean[zona_col]
            else:
                df_clean['zona_numerica'] = df_clean[zona_col].astype(int)
                df_clean['zona_label'] = df_clean[zona_col].apply(lambda x: f'Zona {int(x)}')
        else:
            df_clean['zona_numerica'] = 1
            df_clean['zona_label'] = 'General'
        
        df_sample = df_clean.head(100)
        
        burbujas = []
        for _, row in df_sample.iterrows():
            burbujas.append({
                'x': round(float(row['area_m2']), 2),
                'y': round(float(row['precio']), 2),
                'z': int(row['zona_numerica']),
                'zona': str(row['zona_label'])
            })
        
        return jsonify(burbujas)
    except Exception as e:
        print(f"Error en grafico-burbujas: {str(e)}")
        return jsonify([])

@analisis_bp.route('/datasets/<dataset_id>/analisis-sensibilidad', methods=['GET'])
def obtener_analisis_sensibilidad(dataset_id):
    try:
        dataset = supabase.table('datasets').select('*').eq('id', dataset_id).execute()
        if not dataset.data:
            return jsonify([])
        
        respuesta_archivo = requests.get(dataset.data[0]['archivo_url'], timeout=30)
        respuesta_archivo.raise_for_status()
        df = pd.read_csv(StringIO(respuesta_archivo.text))
        
        if 'precio' not in df.columns:
            return jsonify([])
        
        columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        columnas_numericas = [col for col in columnas_numericas if col != 'precio']
        
        if len(columnas_numericas) == 0:
            return jsonify([])
        
        df_clean = df[columnas_numericas + ['precio']].dropna()
        
        if len(df_clean) < 2:
            return jsonify([])
        
        sensibilidad = []
        for col in columnas_numericas:
            try:
                correlacion = df_clean[col].corr(df_clean['precio'])
                
                if pd.isna(correlacion):
                    correlacion = 0
                else:
                    correlacion = abs(correlacion)
                
                if df_clean[col].std() != 0 and df_clean[col].mean() != 0:
                    coef_variacion = df_clean[col].std() / abs(df_clean[col].mean())
                else:
                    coef_variacion = 0
                
                impacto = correlacion * 100
                volatilidad = min(coef_variacion * 100, 100)
                
                sensibilidad.append({
                    'variable': col.replace('_', ' ').title(),
                    'impacto_precio': round(float(impacto), 2),
                    'volatilidad': round(float(volatilidad), 2),
                    'correlacion': round(float(correlacion), 3)
                })
            except Exception as col_error:
                print(f"Error procesando columna {col}: {str(col_error)}")
                continue
        
        sensibilidad.sort(key=lambda x: x['impacto_precio'], reverse=True)
        
        return jsonify(sensibilidad[:10])
    except Exception as e:
        print(f"Error en analisis-sensibilidad: {str(e)}")
        return jsonify([])

@analisis_bp.route('/datasets/<dataset_id>/clustering', methods=['GET'])
def obtener_clustering(dataset_id):
    try:
        dataset = supabase.table('datasets').select('*').eq('id', dataset_id).execute()
        if not dataset.data:
            return jsonify({'clusters': [], 'centroides': [], 'columnas': []})
        
        respuesta_archivo = requests.get(dataset.data[0]['archivo_url'], timeout=30)
        respuesta_archivo.raise_for_status()
        df = pd.read_csv(StringIO(respuesta_archivo.text))
        
        columnas_clustering = ['area_m2', 'habitaciones', 'precio', 'banos']
        columnas_disponibles = [col for col in columnas_clustering if col in df.columns]
        
        if len(columnas_disponibles) < 2:
            return jsonify({'clusters': [], 'centroides': [], 'columnas': []})
        
        df_clean = df[columnas_disponibles].dropna()
        if len(df_clean) < 10:
            return jsonify({'clusters': [], 'centroides': [], 'columnas': []})
        
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_clean)
        
        n_clusters = min(4, max(2, len(df_clean) // 10))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_clean['cluster'] = kmeans.fit_predict(df_scaled)
        
        clusters = []
        colores = ['#0ea5e9', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981']
        
        for i in range(n_clusters):
            cluster_data = df_clean[df_clean['cluster'] == i]
            
            caracteristicas = {}
            for col in columnas_disponibles:
                caracteristicas[col] = {
                    'promedio': round(float(cluster_data[col].mean()), 2),
                    'min': round(float(cluster_data[col].min()), 2),
                    'max': round(float(cluster_data[col].max()), 2)
                }
            
            clusters.append({
                'id': i,
                'nombre': f'Grupo {i + 1}',
                'cantidad': int(len(cluster_data)),
                'caracteristicas': caracteristicas,
                'color': colores[i % len(colores)]
            })
        
        centroides_scaled = kmeans.cluster_centers_
        centroides = scaler.inverse_transform(centroides_scaled)
        
        centroides_data = []
        for i, centroide in enumerate(centroides):
            centroide_dict = {'cluster': f'Grupo {i + 1}'}
            for j, col in enumerate(columnas_disponibles):
                centroide_dict[col] = round(float(centroide[j]), 2)
            centroides_data.append(centroide_dict)
        
        return jsonify({
            'clusters': clusters,
            'centroides': centroides_data,
            'columnas': columnas_disponibles
        })
    except Exception as e:
        print(f"Error en clustering: {str(e)}")
        return jsonify({'clusters': [], 'centroides': [], 'columnas': []})

@analisis_bp.route('/datasets/<dataset_id>/ranking-caracteristicas-zona', methods=['GET'])
def obtener_ranking_caracteristicas_zona(dataset_id):
    try:
        dataset = supabase.table('datasets').select('*').eq('id', dataset_id).execute()
        if not dataset.data:
            return jsonify([])
        
        respuesta_archivo = requests.get(dataset.data[0]['archivo_url'], timeout=30)
        respuesta_archivo.raise_for_status()
        df = pd.read_csv(StringIO(respuesta_archivo.text))
        
        if 'precio' not in df.columns or 'zona' not in df.columns:
            return jsonify([])
        
        columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        columnas_numericas = [col for col in columnas_numericas if col not in ['precio', 'zona']]
        
        if len(columnas_numericas) == 0:
            return jsonify([])
        
        df_work = df.copy()
        
        zona_mapping = {}
        if df_work['zona'].dtype == 'object':
            unique_zonas = df_work['zona'].unique()
            for idx, zona in enumerate(unique_zonas):
                zona_mapping[zona] = idx
            df_work['zona_numerica'] = df_work['zona'].map(zona_mapping)
            df_work['zona_label'] = df_work['zona']
        else:
            df_work['zona_numerica'] = df_work['zona']
            df_work['zona_label'] = df_work['zona'].apply(lambda x: f'Zona {int(x)}')
        
        df_clean = df_work[columnas_numericas + ['precio', 'zona_numerica', 'zona_label']].dropna()
        
        if len(df_clean) < 10:
            return jsonify([])
        
        zonas = sorted(df_clean['zona_numerica'].unique())
        ranking_zonas = []
        
        for zona in zonas:
            df_zona = df_clean[df_clean['zona_numerica'] == zona]
            
            if len(df_zona) < 5:
                continue
            
            zona_label = df_zona['zona_label'].iloc[0]
            
            X = df_zona[columnas_numericas]
            y = df_zona['precio']
            
            try:
                modelo = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                modelo.fit(X, y)
                
                importancias = modelo.feature_importances_
                
                caracteristicas_zona = []
                for col, imp in zip(columnas_numericas, importancias):
                    caracteristicas_zona.append({
                        'caracteristica': col.replace('_', ' ').title(),
                        'importancia': round(float(imp * 100), 2)
                    })
                
                caracteristicas_zona.sort(key=lambda x: x['importancia'], reverse=True)
                
                ranking_zonas.append({
                    'zona': str(zona_label),
                    'zona_id': int(zona),
                    'total_propiedades': int(len(df_zona)),
                    'precio_promedio': round(float(df_zona['precio'].mean()), 2),
                    'caracteristicas': caracteristicas_zona[:10]
                })
            except Exception as e:
                print(f"Error procesando zona {zona_label}: {str(e)}")
                continue
        
        return jsonify(ranking_zonas)
    except Exception as e:
        print(f"Error en ranking-caracteristicas-zona: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify([])

@analisis_bp.route('/datasets/<dataset_id>/validacion-cruzada', methods=['GET'])
def obtener_validacion_cruzada(dataset_id):
    try:
        dataset = supabase.table('datasets').select('*').eq('id', dataset_id).execute()
        if not dataset.data:
            return jsonify({'scores': [], 'promedio': 0, 'desviacion': 0})
        
        respuesta_archivo = requests.get(dataset.data[0]['archivo_url'], timeout=30)
        respuesta_archivo.raise_for_status()
        df = pd.read_csv(StringIO(respuesta_archivo.text))
        
        if 'precio' not in df.columns:
            return jsonify({'scores': [], 'promedio': 0, 'desviacion': 0})
        
        columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        columnas_numericas = [col for col in columnas_numericas if col != 'precio']
        
        if len(columnas_numericas) == 0:
            return jsonify({'scores': [], 'promedio': 0, 'desviacion': 0})
        
        df_clean = df[columnas_numericas + ['precio']].dropna()
        
        if len(df_clean) < 10:
            return jsonify({'scores': [], 'promedio': 0, 'desviacion': 0})
        
        X = df_clean[columnas_numericas]
        y = df_clean['precio']
        
        modelo_rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        modelo_lr = LinearRegression()
        
        kfold = KFold(n_splits=min(5, len(df_clean) // 10), shuffle=True, random_state=42)
        
        scores_rf = cross_val_score(modelo_rf, X, y, cv=kfold, scoring='r2', n_jobs=-1)
        scores_lr = cross_val_score(modelo_lr, X, y, cv=kfold, scoring='r2', n_jobs=-1)
        
        scores_rf_mse = -cross_val_score(modelo_rf, X, y, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1)
        scores_lr_mse = -cross_val_score(modelo_lr, X, y, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1)
        
        resultados_por_fold = []
        for i in range(len(scores_rf)):
            resultados_por_fold.append({
                'fold': i + 1,
                'random_forest_r2': round(float(scores_rf[i]), 4),
                'linear_regression_r2': round(float(scores_lr[i]), 4),
                'random_forest_mse': round(float(scores_rf_mse[i]), 2),
                'linear_regression_mse': round(float(scores_lr_mse[i]), 2)
            })
        
        return jsonify({
            'scores_por_fold': resultados_por_fold,
            'random_forest': {
                'r2_promedio': round(float(scores_rf.mean()), 4),
                'r2_desviacion': round(float(scores_rf.std()), 4),
                'mse_promedio': round(float(scores_rf_mse.mean()), 2),
                'mse_desviacion': round(float(scores_rf_mse.std()), 2)
            },
            'linear_regression': {
                'r2_promedio': round(float(scores_lr.mean()), 4),
                'r2_desviacion': round(float(scores_lr.std()), 4),
                'mse_promedio': round(float(scores_lr_mse.mean()), 2),
                'mse_desviacion': round(float(scores_lr_mse.std()), 2)
            }
        })
    except Exception as e:
        print(f"Error en validacion-cruzada: {str(e)}")
        return jsonify({'scores': [], 'promedio': 0, 'desviacion': 0})

@analisis_bp.route('/datasets/<dataset_id>/analisis-residuales', methods=['GET'])
def obtener_analisis_residuales(dataset_id):
    try:
        dataset = supabase.table('datasets').select('*').eq('id', dataset_id).execute()
        if not dataset.data:
            return jsonify({})
        
        respuesta_archivo = requests.get(dataset.data[0]['archivo_url'], timeout=30)
        respuesta_archivo.raise_for_status()
        df = pd.read_csv(StringIO(respuesta_archivo.text))
        
        if 'precio' not in df.columns:
            return jsonify({})
        
        columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        columnas_numericas = [col for col in columnas_numericas if col != 'precio']
        
        if len(columnas_numericas) == 0:
            return jsonify({})
        
        df_clean = df[columnas_numericas + ['precio']].dropna()
        
        if len(df_clean) < 10:
            return jsonify({})
        
        X = df_clean[columnas_numericas]
        y = df_clean['precio']
        
        modelo = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        modelo.fit(X, y)
        
        y_pred = modelo.predict(X)
        residuales = y - y_pred
        
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        
        residuales_estandarizados = residuales / residuales.std()
        
        qq_data = []
        residuales_sorted = np.sort(residuales_estandarizados)
        teoricos = np.linspace(-3, 3, len(residuales_sorted))
        step = max(1, len(residuales_sorted) // 100)
        for i in range(0, len(residuales_sorted), step):
            qq_data.append({
                'teorico': round(float(teoricos[i]), 3),
                'observado': round(float(residuales_sorted[i]), 3)
            })
        
        scatter_residuales = []
        step = max(1, len(y_pred) // 200)
        for idx in range(0, len(y_pred), step):
            scatter_residuales.append({
                'predicho': round(float(y_pred[idx]), 2),
                'residual': round(float(residuales.iloc[idx]), 2)
            })
        
        hist_residuales = []
        bins = 15
        counts, bin_edges = np.histogram(residuales, bins=bins)
        for i in range(len(counts)):
            hist_residuales.append({
                'rango': f'{int(bin_edges[i])}',
                'frecuencia': int(counts[i])
            })
        
        outliers = []
        umbral_outlier = 3
        for idx, val in enumerate(residuales_estandarizados):
            if abs(val) > umbral_outlier:
                outliers.append({
                    'indice': int(idx),
                    'residual': round(float(residuales.iloc[idx]), 2),
                    'residual_estandarizado': round(float(val), 3),
                    'valor_real': round(float(y.iloc[idx]), 2),
                    'valor_predicho': round(float(y_pred[idx]), 2)
                })
        
        return jsonify({
            'metricas': {
                'mse': round(float(mse), 2),
                'rmse': round(float(rmse), 2),
                'r2': round(float(r2), 4),
                'residual_promedio': round(float(residuales.mean()), 2),
                'residual_std': round(float(residuales.std()), 2)
            },
            'qq_plot': qq_data,
            'scatter_residuales': scatter_residuales,
            'histograma_residuales': hist_residuales,
            'outliers': outliers[:20],
            'total_outliers': len(outliers)
        })
    except Exception as e:
        print(f"Error en analisis-residuales: {str(e)}")
        return jsonify({})

@analisis_bp.route('/datasets/<dataset_id>/multicolinealidad', methods=['GET'])
def obtener_multicolinealidad(dataset_id):
    try:
        dataset = supabase.table('datasets').select('*').eq('id', dataset_id).execute()
        if not dataset.data:
            return jsonify({})
        
        respuesta_archivo = requests.get(dataset.data[0]['archivo_url'], timeout=30)
        respuesta_archivo.raise_for_status()
        df = pd.read_csv(StringIO(respuesta_archivo.text))
        
        columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(columnas_numericas) < 2:
            return jsonify({})
        
        df_clean = df[columnas_numericas].dropna()
        
        if len(df_clean) < 3:
            return jsonify({})
        
        matriz_correlacion = df_clean.corr()
        
        vif_data = []
        for col in columnas_numericas:
            try:
                X_col = df_clean.drop(columns=[col])
                y_col = df_clean[col]
                
                modelo = LinearRegression()
                modelo.fit(X_col, y_col)
                
                r2 = r2_score(y_col, modelo.predict(X_col))
                
                if r2 < 0.9999:
                    vif = 1 / (1 - r2)
                else:
                    vif = 999.99
                
                if vif > 10:
                    nivel = 'Alto'
                    color = 'danger'
                elif vif > 5:
                    nivel = 'Medio'
                    color = 'warning'
                else:
                    nivel = 'Bajo'
                    color = 'success'
                
                vif_data.append({
                    'variable': col.replace('_', ' ').title(),
                    'vif': round(float(min(vif, 999.99)), 2),
                    'nivel': nivel,
                    'color': color
                })
            except Exception as e:
                print(f"Error calculando VIF para {col}: {str(e)}")
                continue
        
        vif_data.sort(key=lambda x: x['vif'], reverse=True)
        
        pares_alta_correlacion = []
        n = len(columnas_numericas)
        for i in range(n):
            for j in range(i + 1, n):
                corr = matriz_correlacion.iloc[i, j]
                if abs(corr) > 0.7:
                    pares_alta_correlacion.append({
                        'variable1': columnas_numericas[i].replace('_', ' ').title(),
                        'variable2': columnas_numericas[j].replace('_', '').title(),
                        'correlacion': round(float(corr), 3),
                        'abs_correlacion': round(float(abs(corr)), 3)
                    })
        
        pares_alta_correlacion.sort(key=lambda x: x['abs_correlacion'], reverse=True)
        
        matriz_datos = []
        for i, col1 in enumerate(columnas_numericas):
            for j, col2 in enumerate(columnas_numericas):
                matriz_datos.append({
                    'x': col1.replace('_', ' ').title(),
                    'y': col2.replace('_', ' ').title(),
                    'value': round(float(matriz_correlacion.iloc[i, j]), 3)
                })
        
        return jsonify({
            'vif': vif_data,
            'pares_alta_correlacion': pares_alta_correlacion[:15],
            'matriz_correlacion': matriz_datos,
            'variables': [col.replace('_', ' ').title() for col in columnas_numericas],
            'resumen': {
                'total_variables': len(columnas_numericas),
                'variables_alto_vif': len([v for v in vif_data if v['vif'] > 10]),
                'pares_correlacionados': len(pares_alta_correlacion)
            }
        })
    except Exception as e:
        print(f"Error en multicolinealidad: {str(e)}")
        return jsonify({})