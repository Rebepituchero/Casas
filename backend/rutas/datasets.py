from flask import Blueprint, request, jsonify, send_file
import pandas as pd
import requests
import numpy as np
from io import StringIO, BytesIO
from supabase import create_client
from config import Config
from modelos.dataset import GestorDataset
from servicios.limpieza import ServicioLimpieza
import uuid
from datetime import datetime

datasets_bp = Blueprint('datasets', __name__)
supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)

@datasets_bp.route('/datasets', methods=['GET'])
def obtener_datasets():
    respuesta = supabase.table('datasets').select('*').execute()
    return jsonify(respuesta.data)

@datasets_bp.route('/datasets', methods=['POST'])
def crear_dataset():
    datos = request.json
    respuesta_archivo = requests.get(datos['archivo_url'])
    df = pd.read_csv(StringIO(respuesta_archivo.text))
    
    datos_dataset = {
        'nombre': datos['nombre'],
        'archivo_url': datos['archivo_url'],
        'filas': len(df),
        'columnas': len(df.columns),
        'usuario_id': datos['usuario_id'],
        'es_limpio': False
    }
    
    respuesta = supabase.table('datasets').insert(datos_dataset).execute()
    return jsonify(respuesta.data[0])

@datasets_bp.route('/datasets/<dataset_id>', methods=['DELETE'])
def eliminar_dataset(dataset_id):
    try:
        dataset = supabase.table('datasets').select('*').eq('id', dataset_id).execute()
        if dataset.data:
            archivo_url = dataset.data[0]['archivo_url']
            try:
                url_parts = archivo_url.split('/')
                ruta_archivo = '/'.join(url_parts[-2:])
                supabase.storage.from_('datasets').remove([ruta_archivo])
            except:
                pass
        
        supabase.table('experimentos').delete().eq('dataset_id', dataset_id).execute()
        supabase.table('datasets').delete().eq('id', dataset_id).execute()
        
        return jsonify({'mensaje': 'Dataset y experimentos asociados eliminados correctamente'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@datasets_bp.route('/datasets/<dataset_id>/columnas', methods=['GET'])
def obtener_columnas(dataset_id):
    dataset = supabase.table('datasets').select('*').eq('id', dataset_id).execute()
    if not dataset.data:
        return jsonify({'error': 'Dataset no encontrado'}), 404
    
    try:
        respuesta_archivo = requests.get(dataset.data[0]['archivo_url'], timeout=30)
        respuesta_archivo.raise_for_status()
        gestor = GestorDataset(StringIO(respuesta_archivo.text))
        estadisticas = gestor.obtener_estadisticas_columnas()
        return jsonify(estadisticas)
    except Exception as e:
        return jsonify({'error': f'Error al cargar columnas: {str(e)}'}), 500

@datasets_bp.route('/datasets/<dataset_id>/vista-previa', methods=['GET'])
def obtener_vista_previa(dataset_id):
    dataset = supabase.table('datasets').select('*').eq('id', dataset_id).execute()
    if not dataset.data:
        return jsonify({'error': 'Dataset no encontrado'}), 404
    
    try:
        pagina = request.args.get('pagina', 1, type=int)
        filas_por_pagina = 100
        
        respuesta_archivo = requests.get(dataset.data[0]['archivo_url'], timeout=30)
        respuesta_archivo.raise_for_status()
        df = pd.read_csv(StringIO(respuesta_archivo.text))
        
        inicio = (pagina - 1) * filas_por_pagina
        fin = inicio + filas_por_pagina
        
        df_pagina = df.iloc[inicio:fin].copy()
        
        vista_previa_dict = df_pagina.to_dict('records')
        
        result = []
        for record in vista_previa_dict:
            new_record = {}
            for key, value in record.items():
                if pd.isna(value):
                    new_record[key] = '[NULL]'
                elif isinstance(value, float):
                    new_record[key] = round(value, 4)
                else:
                    new_record[key] = str(value)
            result.append(new_record)
        
        total_filas = len(df)
        total_paginas = (total_filas + filas_por_pagina - 1) // filas_por_pagina
        
        return jsonify({
            'datos': result,
            'pagina_actual': pagina,
            'total_paginas': total_paginas,
            'total_filas': total_filas,
            'filas_por_pagina': filas_por_pagina
        })
    except Exception as e:
        return jsonify({'error': f'Error al cargar vista previa: {str(e)}'}), 500

@datasets_bp.route('/datasets/<dataset_id>/correlacion', methods=['GET'])
def obtener_correlacion(dataset_id):
    dataset = supabase.table('datasets').select('*').eq('id', dataset_id).execute()
    if not dataset.data:
        return jsonify({'error': 'Dataset no encontrado'}), 404
    
    try:
        respuesta_archivo = requests.get(dataset.data[0]['archivo_url'], timeout=30)
        respuesta_archivo.raise_for_status()
        df = pd.read_csv(StringIO(respuesta_archivo.text))
        
        columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(columnas_numericas) > 0:
            correlacion = df[columnas_numericas].corr()
            return jsonify({
                'variables': columnas_numericas,
                'matriz': correlacion.values.tolist()
            })
        
        return jsonify({'variables': [], 'matriz': []})
    except Exception as e:
        return jsonify({'error': f'Error al cargar correlación: {str(e)}'}), 500

@datasets_bp.route('/datasets/<dataset_id>/distribucion-clases', methods=['GET'])
def obtener_distribucion_clases(dataset_id):
    dataset = supabase.table('datasets').select('*').eq('id', dataset_id).execute()
    if not dataset.data:
        return jsonify({'error': 'Dataset no encontrado'}), 404
    
    try:
        respuesta_archivo = requests.get(dataset.data[0]['archivo_url'], timeout=30)
        respuesta_archivo.raise_for_status()
        df = pd.read_csv(StringIO(respuesta_archivo.text))
        
        primera_columna = df.columns[0]
        if df[primera_columna].dtype == 'object' or df[primera_columna].nunique() < 20:
            distribucion = df[primera_columna].value_counts(dropna=False)
            result = []
            for clase, cantidad in distribucion.items():
                clase_str = '[NULL]' if pd.isna(clase) else str(clase)
                result.append({'clase': clase_str, 'cantidad': int(cantidad)})
            return jsonify(result)
        
        return jsonify([])
    except Exception as e:
        return jsonify({'error': f'Error al cargar distribución: {str(e)}'}), 500

@datasets_bp.route('/datasets/<dataset_id>/estadisticas', methods=['GET'])
def obtener_estadisticas_datos(dataset_id):
    dataset = supabase.table('datasets').select('*').eq('id', dataset_id).execute()
    if not dataset.data:
        return jsonify({'error': 'Dataset no encontrado'}), 404
    
    try:
        respuesta_archivo = requests.get(dataset.data[0]['archivo_url'], timeout=30)
        respuesta_archivo.raise_for_status()
        gestor = GestorDataset(StringIO(respuesta_archivo.text))
        estadisticas = gestor.obtener_estadisticas_datos()
        return jsonify(estadisticas)
    except Exception as e:
        return jsonify({'error': f'Error al cargar estadísticas: {str(e)}'}), 500

@datasets_bp.route('/datasets/<dataset_id>/mapa-calor', methods=['GET'])
def obtener_mapa_calor(dataset_id):
    try:
        dataset = supabase.table('datasets').select('*').eq('id', dataset_id).execute()
        if not dataset.data:
            return jsonify([])
        
        respuesta_archivo = requests.get(dataset.data[0]['archivo_url'], timeout=30)
        respuesta_archivo.raise_for_status()
        df = pd.read_csv(StringIO(respuesta_archivo.text))
        
        if 'zona' not in df.columns or 'precio' not in df.columns:
            return jsonify([])
        
        df_clean = df[['zona', 'precio']].copy()
        df_clean = df_clean.dropna()
        
        if len(df_clean) == 0:
            return jsonify([])
        
        mapa_calor = df_clean.groupby('zona')['precio'].mean().reset_index()
        mapa_calor.columns = ['zona', 'precio_promedio']
        mapa_calor['zona'] = mapa_calor['zona'].astype(str)
        mapa_calor['precio_promedio'] = mapa_calor['precio_promedio'].round(2)
        
        return jsonify(mapa_calor.to_dict('records'))
        
    except Exception as e:
        return jsonify([])

@datasets_bp.route('/datasets/<dataset_id>/datos-3d', methods=['GET'])
def obtener_datos_3d(dataset_id):
    try:
        dataset = supabase.table('datasets').select('*').eq('id', dataset_id).execute()
        if not dataset.data:
            return jsonify([])
        
        respuesta_archivo = requests.get(dataset.data[0]['archivo_url'], timeout=30)
        respuesta_archivo.raise_for_status()
        df = pd.read_csv(StringIO(respuesta_archivo.text))
        
        if 'area_m2' in df.columns and 'precio' in df.columns and 'habitaciones' in df.columns:
            df_clean = df[['area_m2', 'precio', 'habitaciones']].dropna()
            if len(df_clean) > 0:
                df_sample = df_clean.sample(min(100, len(df_clean)))
                datos_3d = df_sample.to_dict('records')
                for dato in datos_3d:
                    dato['area_m2'] = round(float(dato['area_m2']), 2)
                    dato['precio'] = round(float(dato['precio']), 2)
                    dato['habitaciones'] = int(dato['habitaciones'])
                return jsonify(datos_3d)
        
        return jsonify([])
    except Exception as e:
        return jsonify([])

@datasets_bp.route('/datasets/<dataset_id>/histograma-precio', methods=['GET'])
def obtener_histograma_precio(dataset_id):
    try:
        dataset = supabase.table('datasets').select('*').eq('id', dataset_id).execute()
        if not dataset.data:
            return jsonify([])
        
        respuesta_archivo = requests.get(dataset.data[0]['archivo_url'], timeout=30)
        respuesta_archivo.raise_for_status()
        df = pd.read_csv(StringIO(respuesta_archivo.text))
        
        if 'precio' in df.columns:
            precios = df['precio'].dropna()
            if len(precios) > 0:
                min_precio = precios.min()
                max_precio = precios.max()
                bins = 10
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
                
                return jsonify(histogram)
        
        return jsonify([])
    except Exception as e:
        return jsonify([])

@datasets_bp.route('/datasets/<dataset_id>/boxplot-zonas', methods=['GET'])
def obtener_boxplot_zonas(dataset_id):
    try:
        dataset = supabase.table('datasets').select('*').eq('id', dataset_id).execute()
        if not dataset.data:
            return jsonify([])
        
        respuesta_archivo = requests.get(dataset.data[0]['archivo_url'], timeout=30)
        respuesta_archivo.raise_for_status()
        df = pd.read_csv(StringIO(respuesta_archivo.text))
        
        if 'zona' not in df.columns or 'precio' not in df.columns:
            return jsonify([])
        
        df_clean = df[['zona', 'precio']].dropna()
        
        if len(df_clean) == 0:
            return jsonify([])
        
        boxplot_data = []
        for zona in sorted(df_clean['zona'].unique()):
            precios_zona = df_clean[df_clean['zona'] == zona]['precio']
            if len(precios_zona) > 0:
                q1 = precios_zona.quantile(0.25)
                mediana = precios_zona.median()
                q3 = precios_zona.quantile(0.75)
                min_val = precios_zona.min()
                max_val = precios_zona.max()
                
                boxplot_data.append({
                    'zona': str(zona),
                    'min': round(float(min_val), 2),
                    'q1': round(float(q1), 2),
                    'mediana': round(float(mediana), 2),
                    'q3': round(float(q3), 2),
                    'max': round(float(max_val), 2)
                })
        
        return jsonify(boxplot_data)
        
    except Exception as e:
        return jsonify([])

@datasets_bp.route('/datasets/<dataset_id>/serie-temporal', methods=['GET'])
def obtener_serie_temporal(dataset_id):
    try:
        dataset = supabase.table('datasets').select('*').eq('id', dataset_id).execute()
        if not dataset.data:
            return jsonify([])
        
        respuesta_archivo = requests.get(dataset.data[0]['archivo_url'], timeout=30)
        respuesta_archivo.raise_for_status()
        df = pd.read_csv(StringIO(respuesta_archivo.text))
        
        if 'ano_construccion' in df.columns and 'precio' in df.columns:
            df_clean = df[['ano_construccion', 'precio']].dropna()
            if len(df_clean) > 0:
                serie_temporal = df_clean.groupby('ano_construccion')['precio'].mean().reset_index()
                serie_temporal.columns = ['ano', 'precio_promedio']
                serie_temporal = serie_temporal.sort_values('ano')
                serie_temporal['precio_promedio'] = serie_temporal['precio_promedio'].round(2)
                return jsonify(serie_temporal.to_dict('records'))
        
        return jsonify([])
    except Exception as e:
        return jsonify([])

@datasets_bp.route('/datasets/<dataset_id>/limpiar', methods=['POST'])
def limpiar_dataset(dataset_id):
    try:
        operaciones = request.json
        
        dataset = supabase.table('datasets').select('*').eq('id', dataset_id).execute()
        if not dataset.data:
            return jsonify({'error': 'Dataset no encontrado'}), 404
        
        dataset_info = dataset.data[0]
        respuesta_archivo = requests.get(dataset_info['archivo_url'], timeout=30)
        respuesta_archivo.raise_for_status()
        df = pd.read_csv(StringIO(respuesta_archivo.text))
        
        df_limpio, estadisticas = ServicioLimpieza.aplicar_limpieza(df, operaciones)
        
        csv_buffer = StringIO()
        df_limpio.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode('utf-8')
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        nombre_original = dataset_info['nombre'].replace('.csv', '')
        nombre_limpio = f"{nombre_original}_limpio_{timestamp}.csv"
        ruta_archivo = f"{dataset_info['usuario_id']}/{timestamp}_{nombre_limpio}"
        
        upload_response = supabase.storage.from_('datasets').upload(
            ruta_archivo,
            csv_bytes,
            {'content-type': 'text/csv'}
        )
        
        if hasattr(upload_response, 'error') and upload_response.error:
            return jsonify({'error': f'Error al subir archivo: {upload_response.error}'}), 500
        
        public_url = supabase.storage.from_('datasets').get_public_url(ruta_archivo)
        
        nuevo_dataset_id = str(uuid.uuid4())
        datos_dataset_limpio = {
            'id': nuevo_dataset_id,
            'nombre': nombre_limpio,
            'archivo_url': public_url,
            'filas': len(df_limpio),
            'columnas': len(df_limpio.columns),
            'usuario_id': dataset_info['usuario_id'],
            'es_limpio': True,
            'dataset_original_id': dataset_id
        }
        
        respuesta_insert = supabase.table('datasets').insert(datos_dataset_limpio).execute()
        
        if not respuesta_insert.data:
            return jsonify({'error': 'Error al crear registro del dataset limpio'}), 500
        
        return jsonify({
            'mensaje': 'Limpieza aplicada correctamente',
            'filas_resultantes': len(df_limpio),
            'estadisticas': estadisticas,
            'dataset_limpio_id': nuevo_dataset_id,
            'archivo_url': public_url
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@datasets_bp.route('/datasets/<dataset_id>/descargar', methods=['GET'])
def descargar_dataset(dataset_id):
    try:
        dataset = supabase.table('datasets').select('*').eq('id', dataset_id).execute()
        if not dataset.data:
            return jsonify({'error': 'Dataset no encontrado'}), 404
        
        respuesta_archivo = requests.get(dataset.data[0]['archivo_url'], timeout=30)
        respuesta_archivo.raise_for_status()
        
        return send_file(
            BytesIO(respuesta_archivo.content),
            mimetype='text/csv',
            as_attachment=True,
            download_name=dataset.data[0]['nombre']
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500