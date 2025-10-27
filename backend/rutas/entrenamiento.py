from flask import Blueprint, request, jsonify
import pandas as pd
import requests
from io import StringIO
from supabase import create_client
from config import Config
from modelos.entrenamiento import GestorEntrenamiento
import uuid
from datetime import datetime
import traceback
import pickle
import base64

entrenamiento_bp = Blueprint('entrenamiento', __name__)
supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)

@entrenamiento_bp.route('/entrenamientos', methods=['POST'])
def crear_entrenamiento():
    try:
        configuracion = request.json
        
        if not configuracion.get('dataset_id'):
            return jsonify({'error': 'dataset_id es requerido'}), 400
        
        if not configuracion.get('columnas_entrada') or len(configuracion['columnas_entrada']) == 0:
            return jsonify({'error': 'Debe seleccionar al menos una columna de entrada'}), 400
        
        if not configuracion.get('columna_objetivo'):
            return jsonify({'error': 'Debe seleccionar una columna objetivo'}), 400
        
        dataset = supabase.table('datasets').select('*').eq('id', configuracion['dataset_id']).execute()
        
        if not dataset.data:
            return jsonify({'error': 'Dataset no encontrado'}), 404
        
        respuesta_archivo = requests.get(dataset.data[0]['archivo_url'], timeout=30)
        respuesta_archivo.raise_for_status()
        df = pd.read_csv(StringIO(respuesta_archivo.text))
        
        columnas_faltantes = set(configuracion['columnas_entrada'] + [configuracion['columna_objetivo']]) - set(df.columns)
        if columnas_faltantes:
            return jsonify({'error': f'Columnas no encontradas en el dataset: {", ".join(columnas_faltantes)}'}), 400
        
        experimento_id = str(uuid.uuid4())
        nombre_experimento = f"Modelo_{configuracion['tipo_modelo']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        datos_experimento = {
            'id': experimento_id,
            'nombre': nombre_experimento,
            'dataset_id': configuracion['dataset_id'],
            'configuracion': configuracion,
            'estado': 'entrenando',
            'fecha_creacion': datetime.now().isoformat(),
            'metricas': {},
            'metricas_por_epoca': [],
            'columnas_entrada': configuracion['columnas_entrada'],
            'columna_objetivo': configuracion['columna_objetivo']
        }
        
        insert_response = supabase.table('experimentos').insert(datos_experimento).execute()
        
        if not insert_response.data:
            return jsonify({'error': 'Error al crear experimento en la base de datos'}), 500
        
        try:
            gestor = GestorEntrenamiento(configuracion)
            metricas, metricas_por_epoca = gestor.ejecutar(df)
            
            modelo_serializado = base64.b64encode(pickle.dumps(gestor.modelo)).decode('utf-8')
            scaler_serializado = base64.b64encode(pickle.dumps(gestor.scaler)).decode('utf-8')
            
            datos_actualizacion = {
                'estado': 'completado',
                'metricas': metricas,
                'metricas_por_epoca': metricas_por_epoca if metricas_por_epoca else [],
                'modelo_serializado': modelo_serializado,
                'scaler_serializado': scaler_serializado
            }
            
            if gestor.matriz_confusion is not None:
                datos_actualizacion['matriz_confusion'] = gestor.matriz_confusion
            
            if gestor.curva_roc is not None:
                datos_actualizacion['curva_roc'] = gestor.curva_roc
            
            if gestor.importancia_features is not None:
                datos_actualizacion['importancia_features'] = gestor.importancia_features
            
            if gestor.distribucion_errores is not None:
                datos_actualizacion['distribucion_errores'] = gestor.distribucion_errores
            
            if gestor.predicciones_vs_reales is not None:
                datos_actualizacion['predicciones_vs_reales'] = gestor.predicciones_vs_reales
            
            if gestor.tiempo_por_epoca is not None and len(gestor.tiempo_por_epoca) > 0:
                datos_actualizacion['tiempo_por_epoca'] = gestor.tiempo_por_epoca
            
            update_response = supabase.table('experimentos').update(datos_actualizacion).eq('id', experimento_id).execute()
            
            if not update_response.data:
                return jsonify({'error': 'Error al actualizar el experimento'}), 500
            
            return jsonify({
                'id': experimento_id,
                'estado': 'completado',
                'metricas': metricas,
                'mensaje': 'Entrenamiento completado exitosamente'
            })
            
        except Exception as e:
            error_detalle = traceback.format_exc()
            
            supabase.table('experimentos').update({
                'estado': 'error',
                'error': str(e)
            }).eq('id', experimento_id).execute()
            
            return jsonify({
                'error': str(e),
                'detalle': error_detalle
            }), 500
            
    except Exception as e:
        error_detalle = traceback.format_exc()
        return jsonify({
            'error': str(e),
            'detalle': error_detalle
        }), 500

@entrenamiento_bp.route('/experimentos', methods=['GET'], strict_slashes=False)
@entrenamiento_bp.route('/experimentos/', methods=['GET'], strict_slashes=False)
def obtener_experimentos():
    try:
        respuesta = supabase.table('experimentos').select('*').order('fecha_creacion', desc=True).execute()
        experimentos = respuesta.data if respuesta.data else []
        return jsonify(experimentos), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@entrenamiento_bp.route('/experimentos/<experimento_id>', methods=['GET'])
def obtener_experimento(experimento_id):
    try:
        respuesta = supabase.table('experimentos').select('*').eq('id', experimento_id).execute()
        if not respuesta.data:
            return jsonify({'error': 'Experimento no encontrado'}), 404
        return jsonify(respuesta.data[0])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@entrenamiento_bp.route('/experimentos/<experimento_id>', methods=['DELETE'])
def eliminar_experimento(experimento_id):
    try:
        experimento = supabase.table('experimentos').select('*').eq('id', experimento_id).execute()
        if not experimento.data:
            return jsonify({'error': 'Experimento no encontrado'}), 404
        
        supabase.table('experimentos').delete().eq('id', experimento_id).execute()
        return jsonify({'mensaje': 'Experimento eliminado correctamente'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500