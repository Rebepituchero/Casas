import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'
import type { Dataset, ConfiguracionEntrenamiento } from '../tipos'

export default function Entrenamiento() {
  const navigate = useNavigate()
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [columnas, setColumnas] = useState<string[]>([])
  const [entrenando, setEntrenando] = useState(false)
  const [error, setError] = useState<string>('')
  const [config, setConfig] = useState<ConfiguracionEntrenamiento>({
    dataset_id: '',
    columnas_entrada: [],
    columna_objetivo: '',
    tipo_modelo: 'regresion',
    tasa_aprendizaje: 0.001,
    epocas: 100,
    tamano_lote: 32,
    validacion_split: 0.2,
  })

  useEffect(() => {
    cargarDatasets()
  }, [])

  const cargarDatasets = async () => {
    try {
      const { data } = await axios.get('http://localhost:5000/api/datasets')
      setDatasets(data)
    } catch (error) {
      console.error('Error al cargar datasets:', error)
      setError('Error al cargar datasets')
    }
  }

  const seleccionarDataset = async (datasetId: string) => {
    setConfig({ ...config, dataset_id: datasetId, columnas_entrada: [], columna_objetivo: '' })
    setColumnas([])
    setError('')
    
    if (!datasetId) return
    
    try {
      const { data } = await axios.get(`http://localhost:5000/api/datasets/${datasetId}/columnas`)
      setColumnas(data.map((c: any) => c.nombre))
    } catch (error) {
      console.error('Error al cargar columnas:', error)
      setError('Error al cargar columnas del dataset')
    }
  }

  const iniciarEntrenamiento = async () => {
    setError('')
    
    if (!config.dataset_id) {
      setError('Debe seleccionar un dataset')
      return
    }
    
    if (config.columnas_entrada.length === 0) {
      setError('Debe seleccionar al menos una columna de entrada')
      return
    }
    
    if (!config.columna_objetivo) {
      setError('Debe seleccionar una columna objetivo')
      return
    }
    
    if (config.columnas_entrada.includes(config.columna_objetivo)) {
      setError('La columna objetivo no puede estar en las columnas de entrada')
      return
    }
    
    setEntrenando(true)
    
    try {
      const { data } = await axios.post('http://localhost:5000/api/entrenamientos', config)
      navigate(`/resultados?experimento=${data.id}`)
    } catch (error: any) {
      console.error('Error al iniciar entrenamiento:', error)
      const errorMsg = error.response?.data?.error || 'Error al iniciar el entrenamiento'
      setError(errorMsg)
      setEntrenando(false)
    }
  }

  const toggleColumnaEntrada = (columna: string) => {
    const nuevas = config.columnas_entrada.includes(columna)
      ? config.columnas_entrada.filter((c) => c !== columna)
      : [...config.columnas_entrada, columna]
    setConfig({ ...config, columnas_entrada: nuevas })
  }

  const seleccionarTodasColumnas = () => {
    const columnasDisponibles = columnas.filter(col => col !== config.columna_objetivo)
    setConfig({ ...config, columnas_entrada: columnasDisponibles })
  }

  const deseleccionarTodasColumnas = () => {
    setConfig({ ...config, columnas_entrada: [] })
  }

  return (
    <div className="space-y-8 animate-slide-up">
      <div>
        <h1 className="text-4xl md:text-5xl font-bold text-slate-900 mb-2">Entrenar Modelo de Predicción</h1>
        <p className="text-slate-600 text-lg">Configura el modelo para predecir precio, vida útil y tendencias de casas</p>
      </div>

      {error && (
        <div className="card p-4 bg-rose-50 border-2 border-rose-200">
          <div className="flex items-center space-x-2 text-rose-700">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span className="font-semibold">{error}</span>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-6">
          <div className="card p-6">
            <div className="flex items-center space-x-3 mb-6">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
                </svg>
              </div>
              <h2 className="text-2xl font-bold text-slate-900">Dataset de Casas</h2>
            </div>
            <select
              value={config.dataset_id}
              onChange={(e) => seleccionarDataset(e.target.value)}
              className="input-field"
            >
              <option value="">Seleccionar dataset</option>
              {datasets.map((ds) => (
                <option key={ds.id} value={ds.id}>
                  {ds.nombre}
                </option>
              ))}
            </select>
          </div>

          {columnas.length > 0 && (
            <div className="card p-6">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
                    <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
                    </svg>
                  </div>
                  <h2 className="text-2xl font-bold text-slate-900">Características de Entrada</h2>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={seleccionarTodasColumnas}
                    className="px-3 py-2 text-sm bg-primary-600 hover:bg-primary-700 text-white rounded-lg font-semibold transition-colors"
                    type="button"
                  >
                    Todas
                  </button>
                  <button
                    onClick={deseleccionarTodasColumnas}
                    className="px-3 py-2 text-sm bg-slate-200 hover:bg-slate-300 text-slate-700 rounded-lg font-semibold transition-colors"
                    type="button"
                  >
                    Ninguna
                  </button>
                </div>
              </div>
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {columnas.map((col) => (
                  <label key={col} className="flex items-center space-x-3 cursor-pointer p-3 hover:bg-slate-50 rounded-xl transition-colors group">
                    <input
                      type="checkbox"
                      checked={config.columnas_entrada.includes(col)}
                      onChange={() => toggleColumnaEntrada(col)}
                      disabled={col === config.columna_objetivo}
                      className="w-4 h-4 text-primary-600 rounded focus:ring-primary-500"
                    />
                    <span className={`font-medium group-hover:text-primary-600 transition-colors ${col === config.columna_objetivo ? 'text-slate-400' : 'text-slate-700'}`}>
                      {col}
                    </span>
                  </label>
                ))}
              </div>
            </div>
          )}

          {columnas.length > 0 && (
            <div className="card p-6">
              <div className="flex items-center space-x-3 mb-6">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-500 flex items-center justify-center">
                  <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z" />
                  </svg>
                </div>
                <h2 className="text-2xl font-bold text-slate-900">Variable a Predecir</h2>
              </div>
              <select
                value={config.columna_objetivo}
                onChange={(e) => setConfig({ ...config, columna_objetivo: e.target.value })}
                className="input-field"
              >
                <option value="">Seleccionar variable objetivo</option>
                {columnas.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>

        <div className="space-y-6">
          <div className="card p-6">
            <div className="flex items-center space-x-3 mb-6">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-500 to-orange-500 flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <h2 className="text-2xl font-bold text-slate-900">Tipo de Modelo</h2>
            </div>
            <div className="space-y-3">
              {['regresion', 'red_neuronal'].map((tipo) => (
                <label key={tipo} className="flex items-center space-x-3 cursor-pointer p-4 border-2 border-slate-200 rounded-xl hover:bg-slate-50 hover:border-primary-300 transition-colors group">
                  <input
                    type="radio"
                    name="tipo_modelo"
                    value={tipo}
                    checked={config.tipo_modelo === tipo}
                    onChange={(e) => setConfig({ ...config, tipo_modelo: e.target.value as any })}
                    className="w-4 h-4 text-primary-600"
                  />
                  <div>
                    <span className="font-semibold text-slate-700 capitalize group-hover:text-primary-600 transition-colors block">
                      {tipo === 'regresion' ? 'Regresión (Random Forest)' : 'Red Neuronal Profunda'}
                    </span>
                    <span className="text-xs text-slate-500">
                      {tipo === 'regresion' ? 'Ideal para predicción de precios' : 'Mayor precisión con más datos'}
                    </span>
                  </div>
                </label>
              ))}
            </div>
          </div>

          <div className="card p-6">
            <div className="flex items-center space-x-3 mb-6">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-rose-500 to-pink-500 flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
                </svg>
              </div>
              <h2 className="text-2xl font-bold text-slate-900">Hiperparámetros</h2>
            </div>
            <div className="space-y-5">
              <div>
                <div className="flex justify-between items-center mb-3">
                  <label className="block text-sm font-semibold text-slate-700">Tasa de Aprendizaje</label>
                  <span className="px-3 py-1 bg-primary-100 text-primary-700 rounded-lg text-sm font-bold">{config.tasa_aprendizaje}</span>
                </div>
                <input
                  type="range"
                  min="0.0001"
                  max="0.1"
                  step="0.0001"
                  value={config.tasa_aprendizaje}
                  onChange={(e) => setConfig({ ...config, tasa_aprendizaje: parseFloat(e.target.value) })}
                  className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-primary-600"
                />
              </div>

              <div>
                <label className="block text-sm font-semibold text-slate-700 mb-3">Épocas de Entrenamiento</label>
                <input
                  type="number"
                  value={config.epocas}
                  onChange={(e) => setConfig({ ...config, epocas: parseInt(e.target.value) })}
                  className="input-field"
                  min="1"
                  max="1000"
                />
              </div>

              <div>
                <label className="block text-sm font-semibold text-slate-700 mb-3">Tamaño de Lote</label>
                <input
                  type="number"
                  value={config.tamano_lote}
                  onChange={(e) => setConfig({ ...config, tamano_lote: parseInt(e.target.value) })}
                  className="input-field"
                  min="1"
                  max="256"
                />
              </div>

              <div>
                <div className="flex justify-between items-center mb-3">
                  <label className="block text-sm font-semibold text-slate-700">Split de Validación</label>
                  <span className="px-3 py-1 bg-emerald-100 text-emerald-700 rounded-lg text-sm font-bold">{(config.validacion_split * 100).toFixed(0)}%</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="0.4"
                  step="0.05"
                  value={config.validacion_split}
                  onChange={(e) => setConfig({ ...config, validacion_split: parseFloat(e.target.value) })}
                  className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-emerald-600"
                />
              </div>
            </div>
          </div>

          <button
            onClick={iniciarEntrenamiento}
            disabled={entrenando || !config.dataset_id || config.columnas_entrada.length === 0 || !config.columna_objetivo}
            className="w-full btn-primary text-lg py-4"
          >
            {entrenando ? (
              <div className="flex items-center justify-center space-x-2">
                <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span>Entrenando Modelo...</span>
              </div>
            ) : (
              <div className="flex items-center justify-center space-x-2">
                <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                <span>Iniciar Entrenamiento</span>
              </div>
            )}
          </button>
        </div>
      </div>
    </div>
  )
}