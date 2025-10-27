from flask import Blueprint, jsonify
import pandas as pd
import requests
from io import StringIO
from supabase import create_client
from config import Config
from servicios.analisis_mercado import ServicioAnalisisMercado
import logging

logging.getLogger().setLevel(logging.ERROR)

mercado_bp = Blueprint('mercado', __name__)
supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)

@mercado_bp.route('/datasets/<dataset_id>/segmentacion-mercado', methods=['GET'])
def obtener_segmentacion_mercado(dataset_id):
    try:
        dataset = supabase.table('datasets').select('*').eq('id', dataset_id).execute()
        if not dataset.data:
            return jsonify({'segmentos': [], 'estadisticas': {}})
        
        respuesta_archivo = requests.get(dataset.data[0]['archivo_url'], timeout=30)
        respuesta_archivo.raise_for_status()
        df = pd.read_csv(StringIO(respuesta_archivo.text))
        
        df_segmentado, stats = ServicioAnalisisMercado.segmentar_mercado(df)
        
        distribucion = []
        for segmento, datos in stats.items():
            distribucion.append({
                'segmento': segmento.title(),
                'cantidad': datos['cantidad'],
                'precio_promedio': datos['precio_promedio'],
                'precio_min': datos['precio_min'],
                'precio_max': datos['precio_max'],
                'porcentaje': datos['porcentaje'],
                'area_promedio': datos.get('area_promedio', 0)
            })
        
        return jsonify({
            'distribucion': distribucion,
            'estadisticas': stats,
            'total_propiedades': len(df_segmentado)
        })
    except Exception as e:
        return jsonify({'error': str(e), 'distribucion': [], 'estadisticas': {}}), 500

@mercado_bp.route('/datasets/<dataset_id>/anomalias-precios', methods=['GET'])
def obtener_anomalias_precios(dataset_id):
    try:
        dataset = supabase.table('datasets').select('*').eq('id', dataset_id).execute()
        if not dataset.data:
            return jsonify({'anomalias': [], 'resumen': {}})
        
        respuesta_archivo = requests.get(dataset.data[0]['archivo_url'], timeout=30)
        respuesta_archivo.raise_for_status()
        df = pd.read_csv(StringIO(respuesta_archivo.text))
        
        anomalias, conteo = ServicioAnalisisMercado.detectar_anomalias_precios(df)
        
        gangas = [a for a in anomalias if a['tipo'] == 'ganga']
        sobrevaloradas = [a for a in anomalias if a['tipo'] == 'sobrevalorada']
        
        total = conteo['gangas'] + conteo['sobrevaloradas'] + conteo['normales']
        
        return jsonify({
            'gangas': gangas,
            'sobrevaloradas': sobrevaloradas,
            'resumen': {
                'total_gangas': conteo['gangas'],
                'total_sobrevaloradas': conteo['sobrevaloradas'],
                'total_normales': conteo['normales'],
                'porcentaje_gangas': round(conteo['gangas'] / total * 100, 2) if total > 0 else 0,
                'porcentaje_sobrevaloradas': round(conteo['sobrevaloradas'] / total * 100, 2) if total > 0 else 0
            }
        })
    except Exception as e:
        return jsonify({'error': str(e), 'gangas': [], 'sobrevaloradas': [], 'resumen': {}}), 500

@mercado_bp.route('/datasets/<dataset_id>/score-inversion', methods=['GET'])
def obtener_score_inversion(dataset_id):
    try:
        dataset = supabase.table('datasets').select('*').eq('id', dataset_id).execute()
        if not dataset.data:
            return jsonify({'propiedades': [], 'estadisticas': {}})
        
        respuesta_archivo = requests.get(dataset.data[0]['archivo_url'], timeout=30)
        respuesta_archivo.raise_for_status()
        df = pd.read_csv(StringIO(respuesta_archivo.text))
        
        scores = ServicioAnalisisMercado.calcular_score_inversion(df)
        
        if len(scores) > 0:
            score_promedio = sum(s['score_inversion'] for s in scores) / len(scores)
            excelentes = len([s for s in scores if s['score_inversion'] >= 80])
            muy_buenos = len([s for s in scores if 65 <= s['score_inversion'] < 80])
            buenos = len([s for s in scores if 50 <= s['score_inversion'] < 65])
            regulares = len([s for s in scores if 35 <= s['score_inversion'] < 50])
            bajos = len([s for s in scores if s['score_inversion'] < 35])
        else:
            score_promedio = 0
            excelentes = muy_buenos = buenos = regulares = bajos = 0
        
        return jsonify({
            'propiedades': scores,
            'estadisticas': {
                'score_promedio': round(score_promedio, 2),
                'total_propiedades': len(scores),
                'distribucion': {
                    'excelente': excelentes,
                    'muy_bueno': muy_buenos,
                    'bueno': buenos,
                    'regular': regulares,
                    'bajo': bajos
                }
            }
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'propiedades': [],
            'estadisticas': {
                'score_promedio': 0,
                'total_propiedades': 0,
                'distribucion': {
                    'excelente': 0,
                    'muy_bueno': 0,
                    'bueno': 0,
                    'regular': 0,
                    'bajo': 0
                }
            }
        }), 500

@mercado_bp.route('/datasets/<dataset_id>/analisis-completo-mercado', methods=['GET'])
def obtener_analisis_completo_mercado(dataset_id):
    try:
        dataset = supabase.table('datasets').select('*').eq('id', dataset_id).execute()
        if not dataset.data:
            return jsonify({'error': 'Dataset no encontrado'}), 404
        
        respuesta_archivo = requests.get(dataset.data[0]['archivo_url'], timeout=30)
        respuesta_archivo.raise_for_status()
        df = pd.read_csv(StringIO(respuesta_archivo.text))
        
        df_segmentado, stats_segmentos = ServicioAnalisisMercado.segmentar_mercado(df)
        anomalias, conteo_anomalias = ServicioAnalisisMercado.detectar_anomalias_precios(df)
        scores = ServicioAnalisisMercado.calcular_score_inversion(df)
        
        gangas = [a for a in anomalias if a['tipo'] == 'ganga'][:10]
        sobrevaloradas = [a for a in anomalias if a['tipo'] == 'sobrevalorada'][:10]
        mejores_inversiones = scores[:10]
        
        return jsonify({
            'segmentacion': {
                'estadisticas': stats_segmentos,
                'total': len(df_segmentado)
            },
            'anomalias': {
                'gangas_destacadas': gangas,
                'sobrevaloradas_destacadas': sobrevaloradas,
                'resumen': conteo_anomalias
            },
            'inversion': {
                'mejores_oportunidades': mejores_inversiones,
                'score_promedio': round(sum(s['score_inversion'] for s in scores) / len(scores), 2) if len(scores) > 0 else 0
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500