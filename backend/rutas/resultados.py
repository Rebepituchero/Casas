from flask import Blueprint, jsonify
from supabase import create_client
from config import Config

resultados_bp = Blueprint('resultados', __name__)
supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)

@resultados_bp.route('/experimentos/recientes', methods=['GET'])
def obtener_experimentos_recientes():
    try:
        respuesta = supabase.table('experimentos').select('*').order('fecha_creacion', desc=True).limit(5).execute()
        return jsonify(respuesta.data if respuesta.data else [])
    except Exception as e:
        print(f"Error al obtener experimentos recientes: {str(e)}")
        return jsonify([])

@resultados_bp.route('/experimentos/<experimento_id>/metricas', methods=['GET'])
def obtener_metricas_experimento(experimento_id):
    try:
        experimento = supabase.table('experimentos').select('*').eq('id', experimento_id).execute()
        if not experimento.data:
            return jsonify({'error': 'Experimento no encontrado'}), 404
        return jsonify({
            'metricas': experimento.data[0].get('metricas', {}),
            'metricas_por_epoca': experimento.data[0].get('metricas_por_epoca', [])
        })
    except Exception as e:
        print(f"Error al obtener métricas: {str(e)}")
        return jsonify({'error': str(e)}), 500

@resultados_bp.route('/experimentos/comparacion', methods=['GET'])
def comparar_experimentos():
    try:
        experimentos = supabase.table('experimentos').select('*').eq('estado', 'completado').execute()
        comparacion = []
        for exp in (experimentos.data or []):
            if exp.get('metricas'):
                comparacion.append({
                    'id': exp['id'],
                    'nombre': exp['nombre'],
                    'metricas': exp['metricas'],
                    'fecha_creacion': exp['fecha_creacion']
                })
        return jsonify(comparacion)
    except Exception as e:
        print(f"Error al comparar experimentos: {str(e)}")
        return jsonify([])