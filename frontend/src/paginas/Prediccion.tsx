import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import axios from 'axios'

interface Experimento {
  id: string
  nombre: string
  configuracion: any
  estado: string
  fecha_creacion: string
  metricas?: any
  columnas_entrada: string[]
  columna_objetivo: string
}

interface PrediccionResultado {
  precio_predicho: number
  tiempo_vida_estimado: number
  tendencia_precio: string
  confianza: number
  factores_importantes: Array<{ nombre: string; impacto: number }>
  tiempo_venta: {
    dias_minimo: number
    dias_maximo: number
    nivel_demanda: string
    score_demanda: number
  }
  revalorizacion: {
    tasa_anual: number
    valor_1_ano: number
    incremento_1_ano: number
    valor_3_anos: number
    incremento_3_anos: number
    valor_5_anos: number
    incremento_5_anos: number
    valor_10_anos: number
    incremento_10_anos: number
  }
  rentabilidad_alquiler: {
    ingreso_mensual: number
    ingreso_anual: number
    roi_anual: number
    gastos_anuales: number
    ingreso_neto_anual: number
    anos_recuperacion: number
  }
  riesgo_inversion: {
    score_riesgo: number
    nivel_riesgo: string
    color_riesgo: string
    factores_riesgo: Array<{
      factor: string
      impacto: string
    }>
  }
  costos_mantenimiento: {
    costo_anual: number
    costo_mensual: number
    porcentaje_valor: number
  }
}

export default function Prediccion() {
  const [experimentos, setExperimentos] = useState<Experimento[]>([])
  const [experimentoSeleccionado, setExperimentoSeleccionado] = useState('')
  const [prediciendo, setPrediciendo] = useState(false)
  const [resultado, setResultado] = useState<PrediccionResultado | null>(null)
  const [error, setError] = useState('')
  const [cargando, setCargando] = useState(true)
  const [formData, setFormData] = useState<Record<string, string>>({})
  const [columnasModelo, setColumnasModelo] = useState<string[]>([])

  useEffect(() => {
    cargarExperimentos()
  }, [])

  useEffect(() => {
    if (experimentoSeleccionado) {
      cargarColumnasModelo(experimentoSeleccionado)
    }
  }, [experimentoSeleccionado])

  const cargarExperimentos = async () => {
    setCargando(true)
    try {
      const { data } = await axios.get('http://localhost:5000/api/experimentos')
      const completados = data.filter((exp: Experimento) => exp.estado === 'completado')
      setExperimentos(completados)
      if (completados.length > 0) {
        setExperimentoSeleccionado(completados[0].id)
      }
    } catch (error) {
      console.error('Error al cargar modelos:', error)
      setError('No se pudieron cargar los modelos disponibles')
    } finally {
      setCargando(false)
    }
  }

  const cargarColumnasModelo = (experimentoId: string) => {
    const experimento = experimentos.find(exp => exp.id === experimentoId)
    if (experimento && experimento.columnas_entrada) {
      setColumnasModelo(experimento.columnas_entrada)
      const nuevoFormData: Record<string, string> = {}
      experimento.columnas_entrada.forEach(col => {
        nuevoFormData[col] = ''
      })
      setFormData(nuevoFormData)
    }
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    })
  }

  const obtenerTipoCampo = (nombreColumna: string): string => {
    const nombreLower = nombreColumna.toLowerCase()
    
    if (nombreLower.includes('precio') || nombreLower.includes('costo') || nombreLower.includes('valor')) {
      return 'number'
    }
    if (nombreLower.includes('area') || nombreLower.includes('m2') || nombreLower.includes('superficie')) {
      return 'number'
    }
    if (nombreLower.includes('habitacion') || nombreLower.includes('cuarto') || nombreLower.includes('dormitorio')) {
      return 'number'
    }
    if (nombreLower.includes('bano') || nombreLower.includes('baño')) {
      return 'number'
    }
    if (nombreLower.includes('ano') || nombreLower.includes('año') || nombreLower.includes('fecha')) {
      return 'number'
    }
    if (nombreLower.includes('distancia') || nombreLower.includes('km')) {
      return 'number'
    }
    if (nombreLower.includes('piso') || nombreLower.includes('planta')) {
      return 'number'
    }
    if (nombreLower.includes('garaje') || nombreLower.includes('ascensor') || nombreLower.includes('balcon') ||
        nombreLower.includes('calefaccion') || nombreLower.includes('aire')) {
      return 'select'
    }
    if (nombreLower.includes('estado') || nombreLower.includes('conservacion') || nombreLower.includes('calidad')) {
      return 'select'
    }
    if (nombreLower.includes('orientacion') || nombreLower.includes('direccion')) {
      return 'select'
    }
    if (nombreLower.includes('zona') || nombreLower.includes('region') || nombreLower.includes('distrito')) {
      return 'select'
    }
    
    return 'number'
  }

  const obtenerOpcionesSelect = (nombreColumna: string): Array<{ value: string; label: string }> => {
    const nombreLower = nombreColumna.toLowerCase()
    
    if (nombreLower.includes('garaje') || nombreLower.includes('ascensor') || nombreLower.includes('balcon') ||
        nombreLower.includes('calefaccion') || nombreLower.includes('aire')) {
      return [
        { value: '', label: 'Seleccionar' },
        { value: '0', label: 'No' },
        { value: '1', label: 'Sí' }
      ]
    }
    
    if (nombreLower.includes('estado') || nombreLower.includes('conservacion')) {
      return [
        { value: '', label: 'Seleccionar' },
        { value: '1', label: 'Malo' },
        { value: '2', label: 'Regular' },
        { value: '3', label: 'Bueno' },
        { value: '4', label: 'Excelente' }
      ]
    }
    
    if (nombreLower.includes('orientacion')) {
      return [
        { value: '', label: 'Seleccionar' },
        { value: '1', label: 'Norte' },
        { value: '2', label: 'Sur' },
        { value: '3', label: 'Este' },
        { value: '4', label: 'Oeste' }
      ]
    }
    
    if (nombreLower.includes('zona')) {
      return [
        { value: '', label: 'Seleccionar' },
        { value: '1', label: 'Centro' },
        { value: '2', label: 'Residencial Alta' },
        { value: '3', label: 'Residencial Media' },
        { value: '4', label: 'Periférica' },
        { value: '5', label: 'Rural' }
      ]
    }
    
    return [{ value: '', label: 'Seleccionar' }]
  }

  const obtenerPlaceholder = (nombreColumna: string): string => {
    const nombreLower = nombreColumna.toLowerCase()
    
    if (nombreLower.includes('precio')) return '150000'
    if (nombreLower.includes('area') || nombreLower.includes('m2')) return '100'
    if (nombreLower.includes('habitacion')) return '3'
    if (nombreLower.includes('bano')) return '2'
    if (nombreLower.includes('ano') || nombreLower.includes('año')) return '2015'
    if (nombreLower.includes('distancia')) return '5.5'
    if (nombreLower.includes('piso')) return '3'
    if (nombreLower.includes('planta')) return '5'
    
    return '0'
  }

  const formatearNombreColumna = (nombre: string): string => {
    return nombre
      .replace(/_/g, ' ')
      .replace(/\b\w/g, (char) => char.toUpperCase())
  }

  const realizarPrediccion = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setResultado(null)

    if (!experimentoSeleccionado) {
      setError('Debe seleccionar un modelo entrenado')
      return
    }

    const camposVacios = Object.entries(formData).filter(([, value]) => value === '')
    if (camposVacios.length > 0) {
      setError('Todos los campos son obligatorios')
      return
    }

    setPrediciendo(true)

    try {
      const datosNumericos: Record<string, number> = {}
      Object.entries(formData).forEach(([key, value]) => {
        datosNumericos[key] = parseFloat(value)
      })

      const datosPrediccion = {
        experimento_id: experimentoSeleccionado,
        datos: datosNumericos
      }

      const { data } = await axios.post('http://localhost:5000/api/prediccion', datosPrediccion)
      setResultado(data)
    } catch (error: any) {
      console.error('Error al realizar predicción:', error)
      setError(error.response?.data?.error || 'Error al realizar la predicción')
    } finally {
      setPrediciendo(false)
    }
  }

  const limpiarFormulario = () => {
    const nuevoFormData: Record<string, string> = {}
    columnasModelo.forEach(col => {
      nuevoFormData[col] = ''
    })
    setFormData(nuevoFormData)
    setResultado(null)
    setError('')
  }

  if (cargando) {
    return (
      <div className="space-y-8 animate-slide-up">
        <div className="flex items-center justify-center py-20">
          <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-primary-600"></div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-8 animate-slide-up">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-4xl md:text-5xl font-bold text-slate-900 mb-2">Análisis Completo de Propiedad</h1>
          <p className="text-slate-600 text-lg">Obtén predicciones detalladas sobre precio, demanda, rentabilidad y más</p>
        </div>
        <Link to="/entrenamiento" className="btn-secondary space-x-2">
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
          </svg>
          <span>Entrenar Nuevo Modelo</span>
        </Link>
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

      {experimentos.length === 0 ? (
        <div className="card p-16 text-center">
          <div className="w-32 h-32 mx-auto mb-6 rounded-full bg-gradient-to-br from-slate-100 to-slate-200 flex items-center justify-center">
            <svg className="w-16 h-16 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
          <h3 className="text-2xl font-bold text-slate-900 mb-2">No hay modelos disponibles</h3>
          <p className="text-slate-600 text-lg mb-6">Primero debes entrenar un modelo con datos de casas</p>
          <Link to="/entrenamiento" className="btn-primary inline-flex space-x-2">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            <span>Entrenar Modelo</span>
          </Link>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1 space-y-6">
            <div className="card p-6">
              <div className="flex items-center space-x-3 mb-6">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center">
                  <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                </div>
                <h2 className="text-2xl font-bold text-slate-900">Seleccionar Modelo</h2>
              </div>
              <select
                value={experimentoSeleccionado}
                onChange={(e) => setExperimentoSeleccionado(e.target.value)}
                className="input-field"
              >
                {experimentos.map((exp) => (
                  <option key={exp.id} value={exp.id}>
                    {exp.nombre}
                  </option>
                ))}
              </select>
            </div>

            <form onSubmit={realizarPrediccion} className="card p-6">
              <div className="flex items-center space-x-3 mb-6">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-500 flex items-center justify-center">
                  <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
                  </svg>
                </div>
                <h2 className="text-2xl font-bold text-slate-900">Características</h2>
              </div>

              {columnasModelo.length > 0 ? (
                <div className="space-y-4 max-h-96 overflow-y-auto">
                  {columnasModelo.map((columna) => {
                    const tipoCampo = obtenerTipoCampo(columna)
                    return (
                      <div key={columna}>
                        <label className="block text-sm font-semibold text-slate-700 mb-2">
                          {formatearNombreColumna(columna)}
                        </label>
                        {tipoCampo === 'select' ? (
                          <select
                            name={columna}
                            value={formData[columna] || ''}
                            onChange={handleInputChange}
                            className="input-field"
                          >
                            {obtenerOpcionesSelect(columna).map((opcion) => (
                              <option key={opcion.value} value={opcion.value}>
                                {opcion.label}
                              </option>
                            ))}
                          </select>
                        ) : (
                          <input
                            type="number"
                            name={columna}
                            value={formData[columna] || ''}
                            onChange={handleInputChange}
                            className="input-field"
                            placeholder={obtenerPlaceholder(columna)}
                            step="0.01"
                            min="0"
                          />
                        )}
                      </div>
                    )
                  })}
                </div>
              ) : (
                <div className="text-center py-8 text-slate-600">
                  Seleccione un modelo para ver los campos requeridos
                </div>
              )}

              <div className="flex gap-3 mt-6">
                <button
                  type="submit"
                  disabled={prediciendo || columnasModelo.length === 0}
                  className="flex-1 btn-primary"
                >
                  {prediciendo ? (
                    <div className="flex items-center justify-center space-x-2">
                      <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      <span>Analizando...</span>
                    </div>
                  ) : (
                    <div className="flex items-center justify-center space-x-2">
                      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                      <span>Analizar Propiedad</span>
                    </div>
                  )}
                </button>
                <button
                  type="button"
                  onClick={limpiarFormulario}
                  className="btn-secondary"
                >
                  Limpiar
                </button>
              </div>
            </form>
          </div>

          <div className="lg:col-span-2 space-y-6">
            {resultado ? (
              <>
                <div className="card p-6 bg-gradient-to-br from-emerald-50 to-teal-50 border-2 border-emerald-200">
                  <div className="flex items-center space-x-3 mb-4">
                    <div className="w-12 h-12 rounded-xl bg-emerald-500 flex items-center justify-center">
                      <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </div>
                    <div>
                      <h3 className="text-sm font-semibold text-emerald-700 uppercase">Precio Estimado</h3>
                      <p className="text-3xl font-bold text-emerald-900">
                        ${resultado.precio_predicho.toLocaleString('es-ES', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </p>
                    </div>
                  </div>
                  <div className="pt-4 border-t-2 border-emerald-200">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-emerald-700">Confianza</span>
                      <span className="text-sm font-bold text-emerald-900">{resultado.confianza}%</span>
                    </div>
                    <div className="w-full bg-emerald-200 rounded-full h-2">
                      <div
                        className="bg-emerald-600 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${resultado.confianza}%` }}
                      ></div>
                    </div>
                  </div>
                </div>

                <div className="card p-6">
                  <div className="flex items-center space-x-3 mb-4">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center">
                      <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </div>
                    <h3 className="text-lg font-bold text-slate-900">Tiempo de Venta Estimado</h3>
                  </div>
                  <div className="space-y-4">
                    <div className="p-4 bg-slate-50 rounded-xl">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-semibold text-slate-700">Rango de Días</span>
                        <span className="text-2xl font-bold text-slate-900">
                          {resultado.tiempo_venta.dias_minimo} - {resultado.tiempo_venta.dias_maximo} días
                        </span>
                      </div>
                      <p className="text-xs text-slate-600">Tiempo estimado para concretar la venta</p>
                    </div>
                    <div className="p-4 bg-slate-50 rounded-xl">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-semibold text-slate-700">Nivel de Demanda</span>
                        <span className={`badge ${
                          resultado.tiempo_venta.nivel_demanda === 'Alta' ? 'badge-success' :
                          resultado.tiempo_venta.nivel_demanda === 'Media-Alta' ? 'badge-info' :
                          resultado.tiempo_venta.nivel_demanda === 'Media' ? 'badge-warning' : 'badge-error'
                        }`}>
                          {resultado.tiempo_venta.nivel_demanda}
                        </span>
                      </div>
                      <div className="w-full bg-slate-200 rounded-full h-2 mt-2">
                        <div
                          className={`h-2 rounded-full transition-all duration-500 ${
                            resultado.tiempo_venta.nivel_demanda === 'Alta' ? 'bg-emerald-600' :
                            resultado.tiempo_venta.nivel_demanda === 'Media-Alta' ? 'bg-blue-600' :
                            resultado.tiempo_venta.nivel_demanda === 'Media' ? 'bg-amber-600' : 'bg-rose-600'
                          }`}
                          style={{ width: `${resultado.tiempo_venta.score_demanda}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="card p-6">
                  <div className="flex items-center space-x-3 mb-4">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
                      <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                      </svg>
                    </div>
                    <h3 className="text-lg font-bold text-slate-900">Revalorización Futura</h3>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl border-2 border-purple-200">
                      <div className="text-sm font-semibold text-purple-700 mb-1">En 1 año</div>
                      <div className="text-2xl font-bold text-purple-900">${resultado.revalorizacion.valor_1_ano.toLocaleString()}</div>
                      <div className="text-xs text-purple-600 mt-1">+${resultado.revalorizacion.incremento_1_ano.toLocaleString()}</div>
                    </div>
                    <div className="p-4 bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl border-2 border-purple-200">
                      <div className="text-sm font-semibold text-purple-700 mb-1">En 3 años</div>
                      <div className="text-2xl font-bold text-purple-900">${resultado.revalorizacion.valor_3_anos.toLocaleString()}</div>
                      <div className="text-xs text-purple-600 mt-1">+${resultado.revalorizacion.incremento_3_anos.toLocaleString()}</div>
                    </div>
                    <div className="p-4 bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl border-2 border-purple-200">
                      <div className="text-sm font-semibold text-purple-700 mb-1">En 5 años</div>
                      <div className="text-2xl font-bold text-purple-900">${resultado.revalorizacion.valor_5_anos.toLocaleString()}</div>
                      <div className="text-xs text-purple-600 mt-1">+${resultado.revalorizacion.incremento_5_anos.toLocaleString()}</div>
                    </div>
                    <div className="p-4 bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl border-2 border-purple-200">
                      <div className="text-sm font-semibold text-purple-700 mb-1">En 10 años</div>
                      <div className="text-2xl font-bold text-purple-900">${resultado.revalorizacion.valor_10_anos.toLocaleString()}</div>
                      <div className="text-xs text-purple-600 mt-1">+${resultado.revalorizacion.incremento_10_anos.toLocaleString()}</div>
                    </div>
                  </div>
                  <div className="mt-4 p-3 bg-purple-100 rounded-xl">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-semibold text-purple-700">Tasa de Revalorización Anual</span>
                      <span className="text-lg font-bold text-purple-900">{resultado.revalorizacion.tasa_anual}%</span>
                    </div>
                  </div>
                </div>

                <div className="card p-6">
                  <div className="flex items-center space-x-3 mb-4">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-500 to-orange-500 flex items-center justify-center">
                      <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 9V7a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2m2 4h10a2 2 0 002-2v-6a2 2 0 00-2-2H9a2 2 0 00-2 2v6a2 2 0 002 2zm7-5a2 2 0 11-4 0 2 2 0 014 0z" />
                      </svg>
                    </div>
                    <h3 className="text-lg font-bold text-slate-900">Rentabilidad por Alquiler</h3>
                  </div>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between p-3 bg-slate-50 rounded-xl">
                      <span className="text-sm font-semibold text-slate-700">Ingreso Mensual</span>
                      <span className="text-xl font-bold text-slate-900">${resultado.rentabilidad_alquiler.ingreso_mensual.toLocaleString()}</span>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-slate-50 rounded-xl">
                      <span className="text-sm font-semibold text-slate-700">Ingreso Anual</span>
                      <span className="text-xl font-bold text-slate-900">${resultado.rentabilidad_alquiler.ingreso_anual.toLocaleString()}</span>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-amber-100 rounded-xl">
                      <span className="text-sm font-semibold text-amber-700">ROI Anual</span>
                      <span className="text-2xl font-bold text-amber-900">{resultado.rentabilidad_alquiler.roi_anual.toFixed(2)}%</span>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-slate-50 rounded-xl">
                      <span className="text-sm font-semibold text-slate-700">Gastos Anuales</span>
                      <span className="text-lg font-bold text-rose-600">${resultado.rentabilidad_alquiler.gastos_anuales.toLocaleString()}</span>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-emerald-100 rounded-xl">
                      <span className="text-sm font-semibold text-emerald-700">Ingreso Neto Anual</span>
                      <span className="text-xl font-bold text-emerald-900">${resultado.rentabilidad_alquiler.ingreso_neto_anual.toLocaleString()}</span>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-blue-100 rounded-xl">
                      <span className="text-sm font-semibold text-blue-700">Años para Recuperar Inversión</span>
                      <span className="text-xl font-bold text-blue-900">
                        {resultado.rentabilidad_alquiler.anos_recuperacion < 999 ? `${resultado.rentabilidad_alquiler.anos_recuperacion} años` : 'N/A'}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="card p-6">
                  <div className="flex items-center space-x-3 mb-4">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-rose-500 to-pink-500 flex items-center justify-center">
                      <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                      </svg>
                    </div>
                    <h3 className="text-lg font-bold text-slate-900">Análisis de Riesgo</h3>
                  </div>
                  <div className="space-y-4">
                    <div className={`p-4 rounded-xl border-2 ${
                      resultado.riesgo_inversion.color_riesgo === 'success' ? 'bg-emerald-50 border-emerald-200' :
                      resultado.riesgo_inversion.color_riesgo === 'info' ? 'bg-blue-50 border-blue-200' :
                      resultado.riesgo_inversion.color_riesgo === 'warning' ? 'bg-amber-50 border-amber-200' : 'bg-rose-50 border-rose-200'
                    }`}>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-semibold text-slate-700">Nivel de Riesgo</span>
                        <span className={`text-2xl font-bold ${
                          resultado.riesgo_inversion.color_riesgo === 'success' ? 'text-emerald-900' :
                          resultado.riesgo_inversion.color_riesgo === 'info' ? 'text-blue-900' :
                          resultado.riesgo_inversion.color_riesgo === 'warning' ? 'text-amber-900' : 'text-rose-900'
                        }`}>
                          {resultado.riesgo_inversion.nivel_riesgo}
                        </span>
                      </div>
                      <div className="w-full bg-slate-200 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full transition-all duration-500 ${
                            resultado.riesgo_inversion.color_riesgo === 'success' ? 'bg-emerald-600' :
                            resultado.riesgo_inversion.color_riesgo === 'info' ? 'bg-blue-600' :
                            resultado.riesgo_inversion.color_riesgo === 'warning' ? 'bg-amber-600' : 'bg-rose-600'
                          }`}
                          style={{ width: `${resultado.riesgo_inversion.score_riesgo}%` }}
                        ></div>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="text-sm font-semibold text-slate-700 mb-2">Factores de Riesgo:</div>
                      {resultado.riesgo_inversion.factores_riesgo.map((factor, idx) => (
                        <div key={idx} className="flex items-center justify-between p-3 bg-slate-50 rounded-xl">
                          <span className="text-sm text-slate-700">{factor.factor}</span>
                          <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                            factor.impacto === 'positivo' ? 'bg-emerald-100 text-emerald-700' :
                            factor.impacto === 'negativo' ? 'bg-rose-100 text-rose-700' : 'bg-slate-200 text-slate-700'
                          }`}>
                            {factor.impacto === 'positivo' ? '✓ Positivo' : factor.impacto === 'negativo' ? '✗ Negativo' : '− Neutral'}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="card p-6">
                  <div className="flex items-center space-x-3 mb-4">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center">
                      <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                      </svg>
                    </div>
                    <h3 className="text-lg font-bold text-slate-900">Costos de Mantenimiento</h3>
                  </div>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between p-3 bg-slate-50 rounded-xl">
                      <span className="text-sm font-semibold text-slate-700">Costo Mensual Estimado</span>
                      <span className="text-xl font-bold text-slate-900">${resultado.costos_mantenimiento.costo_mensual.toLocaleString()}</span>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-slate-50 rounded-xl">
                      <span className="text-sm font-semibold text-slate-700">Costo Anual Estimado</span>
                      <span className="text-xl font-bold text-slate-900">${resultado.costos_mantenimiento.costo_anual.toLocaleString()}</span>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-indigo-100 rounded-xl">
                      <span className="text-sm font-semibold text-indigo-700">Porcentaje del Valor</span>
                      <span className="text-xl font-bold text-indigo-900">{resultado.costos_mantenimiento.porcentaje_valor.toFixed(2)}%</span>
                    </div>
                  </div>
                </div>

                <div className="card p-6">
                  <div className="flex items-center space-x-3 mb-4">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center">
                      <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                      </svg>
                    </div>
                    <h3 className="text-lg font-bold text-slate-900">Análisis de Mercado</h3>
                  </div>
                  <div className="space-y-4">
                    <div className="p-4 bg-slate-50 rounded-xl">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-semibold text-slate-700">Tendencia de Precio</span>
                        <span className={`badge ${
                          resultado.tendencia_precio === 'subida' ? 'badge-success' :
                          resultado.tendencia_precio === 'bajada' ? 'badge-error' : 'badge-warning'
                        }`}>
                          {resultado.tendencia_precio === 'subida' ? 'En Alza' :
                           resultado.tendencia_precio === 'bajada' ? 'En Baja' : 'Estable'}
                        </span>
                      </div>
                      <p className="text-xs text-slate-600">
                        {resultado.tendencia_precio === 'subida' ? 'El precio tiende a incrementar en esta zona' :
                         resultado.tendencia_precio === 'bajada' ? 'El precio tiende a disminuir en esta zona' :
                         'El precio se mantiene estable en esta zona'}
                      </p>
                    </div>
                    <div className="p-4 bg-slate-50 rounded-xl">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-semibold text-slate-700">Vida Útil Estimada</span>
                        <span className="text-2xl font-bold text-slate-900">{resultado.tiempo_vida_estimado} años</span>
                      </div>
                      <p className="text-xs text-slate-600">Tiempo estimado de vida útil de la construcción</p>
                    </div>
                  </div>
                </div>

                <div className="card p-6">
                  <div className="flex items-center space-x-3 mb-4">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
                      <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                      </svg>
                    </div>
                    <h3 className="text-lg font-bold text-slate-900">Factores Importantes</h3>
                  </div>
                  <div className="space-y-3">
                    {resultado.factores_importantes.map((factor, idx) => (
                      <div key={idx} className="p-3 bg-slate-50 rounded-xl">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-semibold text-slate-700">{factor.nombre}</span>
                          <span className="text-sm font-bold text-primary-600">{factor.impacto}%</span>
                        </div>
                        <div className="w-full bg-slate-200 rounded-full h-1.5">
                          <div
                            className="bg-primary-600 h-1.5 rounded-full transition-all duration-500"
                            style={{ width: `${factor.impacto}%` }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            ) : (
              <div className="card p-8 text-center">
                <div className="w-20 h-20 mx-auto mb-4 rounded-full bg-gradient-to-br from-slate-100 to-slate-200 flex items-center justify-center">
                  <svg className="w-10 h-10 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
                <h3 className="text-xl font-bold text-slate-900 mb-2">Sin Análisis</h3>
                <p className="text-slate-600">Completa el formulario y presiona "Analizar Propiedad" para obtener predicciones detalladas</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}