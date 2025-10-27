from flask import Flask, request, jsonify
from flask_cors import CORS
from config import Config
from rutas.datasets import datasets_bp
from rutas.entrenamiento import entrenamiento_bp
from rutas.resultados import resultados_bp
from rutas.prediccion import prediccion_bp
from rutas.analisis_avanzado import analisis_bp
from rutas.analisis_mercado import mercado_bp
import logging
import sys

app = Flask(__name__)
app.config.from_object(Config)
app.url_map.strict_slashes = False

logging.getLogger('werkzeug').setLevel(logging.ERROR)
logging.getLogger('flask.app').setLevel(logging.ERROR)

CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

app.register_blueprint(datasets_bp, url_prefix='/api')
app.register_blueprint(entrenamiento_bp, url_prefix='/api')
app.register_blueprint(resultados_bp, url_prefix='/api')
app.register_blueprint(prediccion_bp, url_prefix='/api')
app.register_blueprint(analisis_bp, url_prefix='/api')
app.register_blueprint(mercado_bp, url_prefix='/api')

@app.route('/api/estadisticas', methods=['GET'])
def obtener_estadisticas():
    from supabase import create_client
    supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
    
    try:
        datasets = supabase.table('datasets').select('*').execute()
        experimentos = supabase.table('experimentos').select('*').execute()
        
        experimentos_activos = len([e for e in (experimentos.data or []) if e.get('estado') == 'entrenando'])
        
        ultimo_entrenamiento = None
        if experimentos.data and len(experimentos.data) > 0:
            experimentos_ordenados = sorted(experimentos.data, key=lambda x: x['fecha_creacion'], reverse=True)
            ultimo_entrenamiento = experimentos_ordenados[0]['fecha_creacion']
        
        return jsonify({
            'total_datasets': len(datasets.data or []),
            'total_experimentos': len(experimentos.data or []),
            'experimentos_activos': experimentos_activos,
            'ultimo_entrenamiento': ultimo_entrenamiento
        })
    except Exception as e:
        return jsonify({
            'total_datasets': 0,
            'total_experimentos': 0,
            'experimentos_activos': 0,
            'ultimo_entrenamiento': None
        })

@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({'mensaje': 'API funcionando correctamente'})

if __name__ == '__main__':
    print("\nServidor iniciado en http://localhost:5000\n")
    sys.stdout = open('NUL', 'w') if sys.platform == 'win32' else open('/dev/null', 'w')
    app.run(debug=False, port=5000, host='0.0.0.0', use_reloader=False)