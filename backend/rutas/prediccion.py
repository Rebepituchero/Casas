from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import torch
from supabase import create_client
from config import Config
import requests
from io import StringIO
import pickle
import base64

prediccion_bp = Blueprint('prediccion', __name__)
supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)

@prediccion_bp.route('/prediccion', methods=['POST'])
def realizar_prediccion():
    try:
        datos_request = request.json
        experimento_id = datos_request.get('experimento_id')
        datos_entrada = datos_request.get('datos')
        
        if not experimento_id or not datos_entrada:
            return jsonify({'error': 'experimento_id y datos son requeridos'}), 400
        
        experimento = supabase.table('experimentos').select('*').eq('id', experimento_id).execute()
        
        if not experimento.data:
            return jsonify({'error': 'Experimento no encontrado'}), 404
        
        exp_data = experimento.data[0]
        
        if exp_data.get('estado') != 'completado':
            return jsonify({'error': 'El modelo no ha completado el entrenamiento'}), 400
        
        configuracion = exp_data.get('configuracion', {})
        columnas_entrada = configuracion.get('columnas_entrada', [])
        
        if not columnas_entrada:
            return jsonify({'error': 'No se encontraron columnas de entrada en la configuración'}), 400
        
        datos_prediccion = {}
        for columna in columnas_entrada:
            valor = datos_entrada.get(columna)
            if valor is None:
                return jsonify({'error': f'Falta el valor para la columna: {columna}'}), 400
            datos_prediccion[columna] = float(valor)
        
        df_prediccion = pd.DataFrame([datos_prediccion])
        
        dataset = supabase.table('datasets').select('*').eq('id', configuracion['dataset_id']).execute()
        if not dataset.data:
            return jsonify({'error': 'Dataset no encontrado'}), 404
        
        respuesta_archivo = requests.get(dataset.data[0]['archivo_url'], timeout=30)
        respuesta_archivo.raise_for_status()
        df_train = pd.read_csv(StringIO(respuesta_archivo.text))
        
        for col in columnas_entrada:
            if col in df_train.columns and df_train[col].dtype == 'object':
                if col not in df_prediccion.columns:
                    df_prediccion[col] = df_train[col].mode()[0]
        
        modelo_data = exp_data.get('modelo_serializado')
        scaler_data = exp_data.get('scaler_serializado')
        
        if not modelo_data or not scaler_data:
            return jsonify({'error': 'Modelo no encontrado en el experimento'}), 400
        
        from modelos.entrenamiento import GestorEntrenamiento, RedNeuronal
        from sklearn.preprocessing import StandardScaler
        
        scaler = pickle.loads(base64.b64decode(scaler_data))
        
        X_scaled = scaler.transform(df_prediccion[columnas_entrada].values)
        
        tipo_modelo = configuracion.get('tipo_modelo', 'regresion')
        
        if tipo_modelo == 'red_neuronal':
            modelo_state = pickle.loads(base64.b64decode(modelo_data))
            entrada_dim = X_scaled.shape[1]
            
            if 'output_dim' in modelo_state:
                salida_dim = modelo_state['output_dim']
            else:
                salida_dim = 1
            
            modelo = RedNeuronal(entrada_dim, salida_dim)
            modelo.load_state_dict(modelo_state['state_dict'])
            modelo.eval()
            
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled)
                salida = modelo(X_tensor)
                
                if salida.shape[1] > 1:
                    _, prediccion = torch.max(salida, 1)
                    precio_predicho = float(prediccion.numpy()[0])
                else:
                    precio_predicho = float(salida.numpy()[0][0])
        else:
            modelo = pickle.loads(base64.b64decode(modelo_data))
            precio_predicho = float(modelo.predict(X_scaled)[0])
        
        area_m2 = datos_entrada.get('area_m2', 100)
        habitaciones = datos_entrada.get('habitaciones', 3)
        ano_construccion = datos_entrada.get('ano_construccion', 2015)
        zona = datos_entrada.get('zona', 2)
        estado_conservacion = datos_entrada.get('estado_conservacion', 3)
        
        antiguedad = 2025 - ano_construccion
        tiempo_vida_base = 100
        factor_estado_vida = estado_conservacion * 10
        factor_antiguedad_vida = antiguedad * 0.8
        tiempo_vida_estimado = int(tiempo_vida_base + factor_estado_vida - factor_antiguedad_vida)
        tiempo_vida_estimado = max(30, min(150, tiempo_vida_estimado))
        
        if zona in [1, 2]:
            tendencia = 'subida'
        elif zona >= 4:
            tendencia = 'bajada'
        else:
            tendencia = 'estable'
        
        metricas = exp_data.get('metricas', {})
        if 'r2_score' in metricas:
            confianza = min(95, max(75, metricas['r2_score'] * 100))
        elif 'precision' in metricas:
            confianza = min(95, max(75, metricas['precision'] * 100))
        else:
            confianza = 85.0
        
        importancia_features = exp_data.get('importancia_features', [])
        if importancia_features:
            factores = [
                {'nombre': f['feature'], 'impacto': round(f['importancia'] * 100, 2)}
                for f in importancia_features[:5]
            ]
        else:
            factores = [
                {'nombre': 'Área', 'impacto': 25.0},
                {'nombre': 'Zona', 'impacto': 20.0},
                {'nombre': 'Habitaciones', 'impacto': 15.0},
                {'nombre': 'Estado', 'impacto': 15.0},
                {'nombre': 'Antigüedad', 'impacto': 10.0}
            ]
        
        demanda_score = 50
        if zona in [1, 2]:
            demanda_score += 25
        if area_m2 > 80 and area_m2 < 150:
            demanda_score += 15
        if habitaciones >= 2 and habitaciones <= 4:
            demanda_score += 10
        
        demanda_score = min(100, demanda_score)
        
        if demanda_score >= 80:
            nivel_demanda = 'Alta'
            tiempo_venta_min = 15
            tiempo_venta_max = 30
        elif demanda_score >= 60:
            nivel_demanda = 'Media-Alta'
            tiempo_venta_min = 30
            tiempo_venta_max = 60
        elif demanda_score >= 40:
            nivel_demanda = 'Media'
            tiempo_venta_min = 60
            tiempo_venta_max = 90
        else:
            nivel_demanda = 'Baja'
            tiempo_venta_min = 90
            tiempo_venta_max = 180
        
        tasa_revalorizacion_anual = 0.03
        if zona in [1, 2]:
            tasa_revalorizacion_anual = 0.05
        elif zona == 3:
            tasa_revalorizacion_anual = 0.04
        elif zona == 4:
            tasa_revalorizacion_anual = 0.02
        else:
            tasa_revalorizacion_anual = 0.01
        
        revalorizacion_1_ano = precio_predicho * (1 + tasa_revalorizacion_anual)
        revalorizacion_3_anos = precio_predicho * ((1 + tasa_revalorizacion_anual) ** 3)
        revalorizacion_5_anos = precio_predicho * ((1 + tasa_revalorizacion_anual) ** 5)
        revalorizacion_10_anos = precio_predicho * ((1 + tasa_revalorizacion_anual) ** 10)
        
        precio_alquiler_base = precio_predicho * 0.004
        
        if zona in [1, 2]:
            precio_alquiler_base *= 1.3
        elif zona == 3:
            precio_alquiler_base *= 1.1
        elif zona == 4:
            precio_alquiler_base *= 0.9
        else:
            precio_alquiler_base *= 0.7
        
        if habitaciones == 1:
            precio_alquiler_base *= 0.8
        elif habitaciones == 2:
            precio_alquiler_base *= 1.0
        elif habitaciones == 3:
            precio_alquiler_base *= 1.2
        elif habitaciones >= 4:
            precio_alquiler_base *= 1.4
        
        ingreso_mensual = precio_alquiler_base
        ingreso_anual = ingreso_mensual * 12
        roi_anual = (ingreso_anual / precio_predicho) * 100
        
        gastos_anuales = ingreso_anual * 0.25
        ingreso_neto_anual = ingreso_anual - gastos_anuales
        
        if ingreso_neto_anual > 0:
            anos_recuperacion = precio_predicho / ingreso_neto_anual
        else:
            anos_recuperacion = 999
        
        score_riesgo = 50
        
        if zona in [1, 2]:
            score_riesgo -= 15
        elif zona >= 4:
            score_riesgo += 15
        
        if antiguedad > 30:
            score_riesgo += 15
        elif antiguedad < 5:
            score_riesgo -= 10
        
        if estado_conservacion >= 3:
            score_riesgo -= 10
        else:
            score_riesgo += 10
        
        if roi_anual < 3:
            score_riesgo += 10
        elif roi_anual > 7:
            score_riesgo -= 10
        
        score_riesgo = max(0, min(100, score_riesgo))
        
        if score_riesgo < 30:
            nivel_riesgo = 'Bajo'
            color_riesgo = 'success'
        elif score_riesgo < 50:
            nivel_riesgo = 'Medio-Bajo'
            color_riesgo = 'info'
        elif score_riesgo < 70:
            nivel_riesgo = 'Medio'
            color_riesgo = 'warning'
        else:
            nivel_riesgo = 'Alto'
            color_riesgo = 'danger'
        
        costos_mantenimiento_base = area_m2 * 15
        
        if antiguedad < 5:
            factor_antiguedad = 0.5
        elif antiguedad < 15:
            factor_antiguedad = 1.0
        elif antiguedad < 30:
            factor_antiguedad = 1.5
        else:
            factor_antiguedad = 2.0
        
        costos_mantenimiento = costos_mantenimiento_base * factor_antiguedad
        
        if estado_conservacion == 4:
            costos_mantenimiento *= 0.7
        elif estado_conservacion == 2:
            costos_mantenimiento *= 1.3
        elif estado_conservacion == 1:
            costos_mantenimiento *= 1.6
        
        costos_mensuales = costos_mantenimiento / 12
        
        resultado = {
            'precio_predicho': round(precio_predicho, 2),
            'tiempo_vida_estimado': tiempo_vida_estimado,
            'tendencia_precio': tendencia,
            'confianza': round(confianza, 2),
            'factores_importantes': factores,
            'tiempo_venta': {
                'dias_minimo': tiempo_venta_min,
                'dias_maximo': tiempo_venta_max,
                'nivel_demanda': nivel_demanda,
                'score_demanda': round(demanda_score, 2)
            },
            'revalorizacion': {
                'tasa_anual': round(tasa_revalorizacion_anual * 100, 2),
                'valor_1_ano': round(revalorizacion_1_ano, 2),
                'incremento_1_ano': round(revalorizacion_1_ano - precio_predicho, 2),
                'valor_3_anos': round(revalorizacion_3_anos, 2),
                'incremento_3_anos': round(revalorizacion_3_anos - precio_predicho, 2),
                'valor_5_anos': round(revalorizacion_5_anos, 2),
                'incremento_5_anos': round(revalorizacion_5_anos - precio_predicho, 2),
                'valor_10_anos': round(revalorizacion_10_anos, 2),
                'incremento_10_anos': round(revalorizacion_10_anos - precio_predicho, 2)
            },
            'rentabilidad_alquiler': {
                'ingreso_mensual': round(ingreso_mensual, 2),
                'ingreso_anual': round(ingreso_anual, 2),
                'roi_anual': round(roi_anual, 2),
                'gastos_anuales': round(gastos_anuales, 2),
                'ingreso_neto_anual': round(ingreso_neto_anual, 2),
                'anos_recuperacion': round(anos_recuperacion, 1) if anos_recuperacion < 999 else 999
            },
            'riesgo_inversion': {
                'score_riesgo': round(score_riesgo, 2),
                'nivel_riesgo': nivel_riesgo,
                'color_riesgo': color_riesgo,
                'factores_riesgo': [
                    {
                        'factor': 'Ubicación',
                        'impacto': 'positivo' if zona in [1, 2] else 'negativo' if zona >= 4 else 'neutral'
                    },
                    {
                        'factor': 'Antigüedad',
                        'impacto': 'positivo' if antiguedad < 10 else 'negativo' if antiguedad > 30 else 'neutral'
                    },
                    {
                        'factor': 'Estado',
                        'impacto': 'positivo' if estado_conservacion >= 3 else 'negativo'
                    }
                ]
            },
            'costos_mantenimiento': {
                'costo_anual': round(costos_mantenimiento, 2),
                'costo_mensual': round(costos_mensuales, 2),
                'porcentaje_valor': round((costos_mantenimiento / precio_predicho) * 100, 2)
            }
        }
        
        return jsonify(resultado)
        
    except Exception as e:
        print(f"Error en predicción: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500