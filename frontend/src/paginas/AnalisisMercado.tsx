import { useState, useEffect } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import axios from 'axios'
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer} from 'recharts'

interface SegmentoMercado {
  segmento: string
  cantidad: number
  precio_promedio: number
  precio_min: number
  precio_max: number
  porcentaje: number
  area_promedio: number
}

interface Anomalia {
  indice: number
  precio: number
  tipo: string
  score: number
  desviacion_media: number
  area_m2?: number
  habitaciones?: number
}

interface ScoreInversion {
  indice: number
  precio: number
  score_inversion: number
  potencial: string
  area_m2?: number
  zona?: number
  ano_construccion?: number
}

export default function AnalisisMercado() {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const datasetId = searchParams.get('dataset')
  
  const [segmentos, setSegmentos] = useState<SegmentoMercado[]>([])
  const [gangas, setGangas] = useState<Anomalia[]>([])
  const [sobrevaloradas, setSobrevaloradas] = useState<Anomalia[]>([])
  const [scoresInversion, setScoresInversion] = useState<ScoreInversion[]>([])
  const [resumenAnomalias, setResumenAnomalias] = useState<any>(null)
  const [estadisticasInversion, setEstadisticasInversion] = useState<any>(null)
  const [cargando, setCargando] = useState(true)
  const [vistaActiva, setVistaActiva] = useState<'segmentacion' | 'anomalias' | 'inversion'>('segmentacion')

  const COLORES_SEGMENTOS = {
    'Economico': '#10b981',
    'Medio': '#f59e0b',
    'Lujo': '#8b5cf6'
  }

  const COLORES_POTENCIAL = {
    'Excelente': '#10b981',
    'Muy Bueno': '#14b8a6',
    'Bueno': '#f59e0b',
    'Regular': '#f97316',
    'Bajo': '#ef4444'
  }

  useEffect(() => {
    if (datasetId) {
      cargarDatos()
    }
  }, [datasetId])

  const cargarDatos = async () => {
    setCargando(true)
    try {
      const [segmentacion, anomalias, inversion] = await Promise.all([
        axios.get(`http://localhost:5000/api/datasets/${datasetId}/segmentacion-mercado`).catch(() => ({ data: { distribucion: [], estadisticas: {} } })),
        axios.get(`http://localhost:5000/api/datasets/${datasetId}/anomalias-precios`).catch(() => ({ data: { gangas: [], sobrevaloradas: [], resumen: {} } })),
        axios.get(`http://localhost:5000/api/datasets/${datasetId}/score-inversion`).catch(() => ({ data: { propiedades: [], estadisticas: {} } }))
      ])

      setSegmentos(segmentacion.data.distribucion || [])
      setGangas(anomalias.data.gangas || [])
      setSobrevaloradas(anomalias.data.sobrevaloradas || [])
      setResumenAnomalias(anomalias.data.resumen || {})
      setScoresInversion(inversion.data.propiedades || [])
      setEstadisticasInversion(inversion.data.estadisticas || {})
    } catch (error) {
      console.error('Error al cargar datos:', error)
    } finally {
      setCargando(false)
    }
  }

  if (!datasetId) {
    return (
      <div className="space-y-8 animate-slide-up">
        <div className="card p-16 text-center">
          <div className="w-32 h-32 mx-auto mb-6 rounded-full bg-gradient-to-br from-slate-100 to-slate-200 flex items-center justify-center">
            <svg className="w-16 h-16 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <h3 className="text-2xl font-bold text-slate-900 mb-2">Selecciona un Dataset</h3>
          <p className="text-slate-600 text-lg mb-6">Para ver el análisis de mercado</p>
          <button onClick={() => navigate('/datasets')} className="btn-primary inline-flex space-x-2">
            <span>Ir a Datasets</span>
          </button>
        </div>
      </div>
    )
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
          <h1 className="text-4xl md:text-5xl font-bold text-slate-900 mb-2">Análisis de Mercado</h1>
          <p className="text-slate-600 text-lg">Segmentación, anomalías y oportunidades de inversión</p>
        </div>
        <button onClick={() => navigate('/datasets')} className="btn-secondary space-x-2">
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
          <span>Volver a Datasets</span>
        </button>
      </div>

      <div className="flex gap-2 overflow-x-auto pb-2">
        <button
          onClick={() => setVistaActiva('segmentacion')}
          className={`px-4 py-2 rounded-xl font-semibold transition-all whitespace-nowrap flex items-center space-x-2 ${
            vistaActiva === 'segmentacion'
              ? 'bg-primary-600 text-white shadow-lg'
              : 'bg-white text-slate-700 border-2 border-slate-200'
          }`}
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z" />
          </svg>
          <span>Segmentación</span>
        </button>
        <button
          onClick={() => setVistaActiva('anomalias')}
          className={`px-4 py-2 rounded-xl font-semibold transition-all whitespace-nowrap flex items-center space-x-2 ${
            vistaActiva === 'anomalias'
              ? 'bg-primary-600 text-white shadow-lg'
              : 'bg-white text-slate-700 border-2 border-slate-200'
          }`}
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <span>Anomalías</span>
        </button>
        <button
          onClick={() => setVistaActiva('inversion')}
          className={`px-4 py-2 rounded-xl font-semibold transition-all whitespace-nowrap flex items-center space-x-2 ${
            vistaActiva === 'inversion'
              ? 'bg-primary-600 text-white shadow-lg'
              : 'bg-white text-slate-700 border-2 border-slate-200'
          }`}
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
          </svg>
          <span>Score Inversión</span>
        </button>
      </div>

      {vistaActiva === 'segmentacion' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="card p-8">
              <h3 className="text-2xl font-bold text-slate-900 mb-6 flex items-center space-x-2">
                <svg className="w-7 h-7 text-primary-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z" />
                </svg>
                <span>Distribución por Segmento</span>
              </h3>
              {segmentos.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={segmentos}
                      dataKey="cantidad"
                      nameKey="segmento"
                      cx="50%"
                      cy="50%"
                      outerRadius={100}
                      label={({ segmento, porcentaje }) => `${segmento}: ${porcentaje}%`}
                    >
                      {segmentos.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORES_SEGMENTOS[entry.segmento as keyof typeof COLORES_SEGMENTOS]} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '2px solid #e2e8f0', borderRadius: '12px' }} />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <div className="text-center py-12 text-slate-600">No hay datos de segmentación</div>
              )}
            </div>

            <div className="card p-8">
              <h3 className="text-2xl font-bold text-slate-900 mb-6 flex items-center space-x-2">
                <svg className="w-7 h-7 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                <span>Precios por Segmento</span>
              </h3>
              {segmentos.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={segmentos}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="segmento" stroke="#64748b" />
                    <YAxis stroke="#64748b" />
                    <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '2px solid #e2e8f0', borderRadius: '12px' }} />
                    <Legend />
                    <Bar dataKey="precio_promedio" fill="#0ea5e9" name="Precio Promedio" radius={[8, 8, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="text-center py-12 text-slate-600">No hay datos de precios</div>
              )}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {segmentos.map((segmento, idx) => (
              <div key={idx} className="card p-6" style={{ borderTop: `4px solid ${COLORES_SEGMENTOS[segmento.segmento as keyof typeof COLORES_SEGMENTOS]}` }}>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-xl font-bold text-slate-900">{segmento.segmento}</h3>
                  <div className="px-3 py-1 rounded-full text-white font-bold text-sm" style={{ backgroundColor: COLORES_SEGMENTOS[segmento.segmento as keyof typeof COLORES_SEGMENTOS] }}>
                    {segmento.porcentaje}%
                  </div>
                </div>
                <div className="space-y-3">
                  <div className="flex justify-between items-center p-3 bg-slate-50 rounded-lg">
                    <span className="text-sm text-slate-600">Cantidad:</span>
                    <span className="font-bold text-slate-900">{segmento.cantidad}</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-slate-50 rounded-lg">
                    <span className="text-sm text-slate-600">Precio Promedio:</span>
                    <span className="font-bold text-slate-900">${segmento.precio_promedio.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-slate-50 rounded-lg">
                    <span className="text-sm text-slate-600">Rango:</span>
                    <span className="font-bold text-slate-900">${segmento.precio_min.toLocaleString()} - ${segmento.precio_max.toLocaleString()}</span>
                  </div>
                  {segmento.area_promedio > 0 && (
                    <div className="flex justify-between items-center p-3 bg-slate-50 rounded-lg">
                      <span className="text-sm text-slate-600">Área Promedio:</span>
                      <span className="font-bold text-slate-900">{segmento.area_promedio} m²</span>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {vistaActiva === 'anomalias' && (
        <div className="space-y-6">
          {resumenAnomalias && (
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="card p-6 bg-gradient-to-br from-emerald-50 to-teal-50 border-2 border-emerald-200">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-semibold text-emerald-700 uppercase">Gangas</span>
                  <svg className="w-6 h-6 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                  </svg>
                </div>
                <p className="text-3xl font-bold text-emerald-900">{resumenAnomalias.total_gangas}</p>
                <p className="text-sm text-emerald-700 mt-1">{resumenAnomalias.porcentaje_gangas}% del total</p>
              </div>
              <div className="card p-6 bg-gradient-to-br from-rose-50 to-pink-50 border-2 border-rose-200">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-semibold text-rose-700 uppercase">Sobrevaloradas</span>
                  <svg className="w-6 h-6 text-rose-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
                  </svg>
                </div>
                <p className="text-3xl font-bold text-rose-900">{resumenAnomalias.total_sobrevaloradas}</p>
                <p className="text-sm text-rose-700 mt-1">{resumenAnomalias.porcentaje_sobrevaloradas}% del total</p>
              </div>
              <div className="card p-6 bg-gradient-to-br from-blue-50 to-indigo-50 border-2 border-blue-200">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-semibold text-blue-700 uppercase">Normales</span>
                  <svg className="w-6 h-6 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <p className="text-3xl font-bold text-blue-900">{resumenAnomalias.total_normales}</p>
                <p className="text-sm text-blue-700 mt-1">Precios equilibrados</p>
              </div>
              <div className="card p-6 bg-gradient-to-br from-purple-50 to-violet-50 border-2 border-purple-200">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-semibold text-purple-700 uppercase">Total Analizadas</span>
                  <svg className="w-6 h-6 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                </div>
                <p className="text-3xl font-bold text-purple-900">{resumenAnomalias.total_gangas + resumenAnomalias.total_sobrevaloradas + resumenAnomalias.total_normales}</p>
                <p className="text-sm text-purple-700 mt-1">Propiedades totales</p>
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="card p-6">
              <h3 className="text-2xl font-bold text-slate-900 mb-6 flex items-center space-x-2">
                <svg className="w-7 h-7 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span>Mejores Gangas</span>
              </h3>
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {gangas.length > 0 ? (
                  gangas.slice(0, 10).map((ganga, idx) => (
                    <div key={idx} className="p-4 bg-gradient-to-r from-emerald-50 to-teal-50 rounded-xl border-2 border-emerald-200 hover:shadow-lg transition-shadow">
                      <div className="flex justify-between items-start mb-2">
                        <div className="flex items-center space-x-2">
                          <div className="w-8 h-8 rounded-full bg-emerald-600 text-white flex items-center justify-center font-bold text-sm">
                            {idx + 1}
                          </div>
                          <span className="text-lg font-bold text-slate-900">${ganga.precio.toLocaleString()}</span>
                        </div>
                        <span className="px-3 py-1 bg-emerald-600 text-white rounded-full text-xs font-bold">
                          {ganga.desviacion_media}% bajo media
                        </span>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        {ganga.area_m2 && (
                          <div className="flex justify-between">
                            <span className="text-slate-600">Área:</span>
                            <span className="font-semibold text-slate-900">{ganga.area_m2} m²</span>
                          </div>
                        )}
                        {ganga.habitaciones && (
                          <div className="flex justify-between">
                            <span className="text-slate-600">Habitaciones:</span>
                            <span className="font-semibold text-slate-900">{ganga.habitaciones}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-12 text-slate-600">No se detectaron gangas</div>
                )}
              </div>
            </div>

            <div className="card p-6">
              <h3 className="text-2xl font-bold text-slate-900 mb-6 flex items-center space-x-2">
                <svg className="w-7 h-7 text-rose-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                <span>Propiedades Sobrevaloradas</span>
              </h3>
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {sobrevaloradas.length > 0 ? (
                  sobrevaloradas.slice(0, 10).map((prop, idx) => (
                    <div key={idx} className="p-4 bg-gradient-to-r from-rose-50 to-pink-50 rounded-xl border-2 border-rose-200 hover:shadow-lg transition-shadow">
                      <div className="flex justify-between items-start mb-2">
                        <div className="flex items-center space-x-2">
                          <div className="w-8 h-8 rounded-full bg-rose-600 text-white flex items-center justify-center font-bold text-sm">
                            {idx + 1}
                          </div>
                          <span className="text-lg font-bold text-slate-900">${prop.precio.toLocaleString()}</span>
                        </div>
                        <span className="px-3 py-1 bg-rose-600 text-white rounded-full text-xs font-bold">
                          +{prop.desviacion_media}% sobre media
                        </span>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        {prop.area_m2 && (
                          <div className="flex justify-between">
                            <span className="text-slate-600">Área:</span>
                            <span className="font-semibold text-slate-900">{prop.area_m2} m²</span>
                          </div>
                        )}
                        {prop.habitaciones && (
                          <div className="flex justify-between">
                            <span className="text-slate-600">Habitaciones:</span>
                            <span className="font-semibold text-slate-900">{prop.habitaciones}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-12 text-slate-600">No se detectaron sobrevaloradas</div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {vistaActiva === 'inversion' && (
        <div className="space-y-6">
          {estadisticasInversion && estadisticasInversion.distribucion && (
            <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
              <div className="card p-6 bg-gradient-to-br from-emerald-50 to-green-50 border-2 border-emerald-200">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-semibold text-emerald-700 uppercase">Excelente</span>
                  <div className="w-3 h-3 rounded-full bg-emerald-600"></div>
                </div>
                <p className="text-3xl font-bold text-emerald-900">{estadisticasInversion.distribucion.excelente}</p>
                <p className="text-xs text-emerald-700 mt-1">Score ≥ 80</p>
              </div>
              <div className="card p-6 bg-gradient-to-br from-teal-50 to-cyan-50 border-2 border-teal-200">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-semibold text-teal-700 uppercase">Muy Bueno</span>
                  <div className="w-3 h-3 rounded-full bg-teal-600"></div>
                </div>
                <p className="text-3xl font-bold text-teal-900">{estadisticasInversion.distribucion.muy_bueno}</p>
                <p className="text-xs text-teal-700 mt-1">Score 65-79</p>
              </div>
              <div className="card p-6 bg-gradient-to-br from-amber-50 to-yellow-50 border-2 border-amber-200">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-semibold text-amber-700 uppercase">Bueno</span>
                  <div className="w-3 h-3 rounded-full bg-amber-600"></div>
                </div>
                <p className="text-3xl font-bold text-amber-900">{estadisticasInversion.distribucion.bueno}</p>
                <p className="text-xs text-amber-700 mt-1">Score 50-64</p>
              </div>
              <div className="card p-6 bg-gradient-to-br from-orange-50 to-red-50 border-2 border-orange-200">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-semibold text-orange-700 uppercase">Regular</span>
                  <div className="w-3 h-3 rounded-full bg-orange-600"></div>
                </div>
                <p className="text-3xl font-bold text-orange-900">{estadisticasInversion.distribucion.regular}</p>
                <p className="text-xs text-orange-700 mt-1">Score 35-49</p>
              </div>
              <div className="card p-6 bg-gradient-to-br from-rose-50 to-pink-50 border-2 border-rose-200">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-semibold text-rose-700 uppercase">Bajo</span>
                  <div className="w-3 h-3 rounded-full bg-rose-600"></div>
                </div>
                <p className="text-3xl font-bold text-rose-900">{estadisticasInversion.distribucion.bajo}</p>
                <p className="text-xs text-rose-700 mt-1">Score &lt; 35</p>
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="card p-8">
              <h3 className="text-2xl font-bold text-slate-900 mb-6 flex items-center space-x-2">
                <svg className="w-7 h-7 text-primary-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z" />
                </svg>
                <span>Distribución Score de Inversión</span>
              </h3>
              {estadisticasInversion && estadisticasInversion.distribucion && (
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={[
                        { name: 'Excelente', value: estadisticasInversion.distribucion.excelente },
                        { name: 'Muy Bueno', value: estadisticasInversion.distribucion.muy_bueno },
                        { name: 'Bueno', value: estadisticasInversion.distribucion.bueno },
                        { name: 'Regular', value: estadisticasInversion.distribucion.regular },
                        { name: 'Bajo', value: estadisticasInversion.distribucion.bajo }
                      ]}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={100}
                      label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    >
                      {Object.entries(COLORES_POTENCIAL).map(([, color], index) => (
                        <Cell key={`cell-${index}`} fill={color} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '2px solid #e2e8f0', borderRadius: '12px' }} />
                  </PieChart>
                </ResponsiveContainer>
              )}
            </div>

            <div className="card p-8">
              <h3 className="text-2xl font-bold text-slate-900 mb-6 flex items-center space-x-2">
                <svg className="w-7 h-7 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                <span>Score Promedio por Categoría</span>
              </h3>
              {estadisticasInversion && (
                <div className="space-y-4">
                  <div className="p-4 bg-gradient-to-r from-emerald-50 to-teal-50 rounded-xl border-2 border-emerald-200">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-semibold text-emerald-700">Score Promedio General</span>
                      <span className="text-2xl font-bold text-emerald-900">{estadisticasInversion.score_promedio}</span>
                    </div>
                    <div className="w-full bg-emerald-200 rounded-full h-3">
                      <div className="bg-emerald-600 h-3 rounded-full transition-all duration-500" style={{ width: `${estadisticasInversion.score_promedio}%` }}></div>
                    </div>
                  </div>
                  <div className="p-4 bg-slate-50 rounded-xl">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-semibold text-slate-700">Total Propiedades Analizadas</span>
                      <span className="text-2xl font-bold text-slate-900">{estadisticasInversion.total_propiedades}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="card p-6">
            <h3 className="text-2xl font-bold text-slate-900 mb-6 flex items-center space-x-2">
              <svg className="w-7 h-7 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
              </svg>
              <span>Top 20 Mejores Oportunidades de Inversión</span>
            </h3>
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {scoresInversion.length > 0 ? (
                scoresInversion.slice(0, 20).map((prop, idx) => (
                  <div key={idx} className="p-5 bg-gradient-to-r from-slate-50 to-slate-100 rounded-xl border-2 border-slate-200 hover:shadow-lg transition-all hover:scale-102">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-3">
                        <div className="w-10 h-10 rounded-full flex items-center justify-center font-bold text-white" style={{ backgroundColor: COLORES_POTENCIAL[prop.potencial as keyof typeof COLORES_POTENCIAL] }}>
                          {idx + 1}
                        </div>
                        <div>
                          <div className="text-lg font-bold text-slate-900">${prop.precio.toLocaleString()}</div>
                          {prop.area_m2 && (
                            <div className="text-sm text-slate-600">{prop.area_m2} m²</div>
                          )}
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-2xl font-bold text-slate-900">{prop.score_inversion}</div>
                        <div className="px-3 py-1 rounded-full text-xs font-bold text-white" style={{ backgroundColor: COLORES_POTENCIAL[prop.potencial as keyof typeof COLORES_POTENCIAL] }}>
                          {prop.potencial}
                        </div>
                      </div>
                    </div>
                    <div className="w-full bg-slate-200 rounded-full h-2">
                      <div className="h-2 rounded-full transition-all duration-500" style={{ width: `${prop.score_inversion}%`, backgroundColor: COLORES_POTENCIAL[prop.potencial as keyof typeof COLORES_POTENCIAL] }}></div>
                    </div>
                    {(prop.zona || prop.ano_construccion) && (
                      <div className="grid grid-cols-2 gap-2 mt-3 text-sm">
                        {prop.zona && (
                          <div className="flex justify-between p-2 bg-white rounded-lg">
                            <span className="text-slate-600">Zona:</span>
                            <span className="font-semibold text-slate-900">{prop.zona}</span>
                          </div>
                        )}
                        {prop.ano_construccion && (
                          <div className="flex justify-between p-2 bg-white rounded-lg">
                            <span className="text-slate-600">Año:</span>
                            <span className="font-semibold text-slate-900">{prop.ano_construccion}</span>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))
              ) : (
                <div className="text-center py-12 text-slate-600">No hay datos de inversión disponibles</div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}