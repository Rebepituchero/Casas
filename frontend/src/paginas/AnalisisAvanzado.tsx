import { useState, useEffect } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import axios from 'axios'
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend, ResponsiveContainer, Tooltip, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, ZAxis, BarChart, Bar, Cell, LineChart, Line } from 'recharts'

interface RadarData {
  casa: string
  metricas: Array<{ metrica: string; valor: number }>
}

interface SankeyData {
  nodes: Array<{ id: number; name: string }>
  links: Array<{ source: number; target: number; value: number }>
}

interface BurbujaData {
  x: number
  y: number
  z: number
  zona: string
}

interface SensibilidadData {
  variable: string
  impacto_precio: number
  volatilidad: number
  correlacion: number
}

interface ClusterData {
  clusters: Array<{
    id: number
    nombre: string
    cantidad: number
    caracteristicas: any
    color: string
  }>
  centroides: any[]
  columnas: string[]
}

interface RankingZona {
  zona: string
  zona_id: number
  total_propiedades: number
  precio_promedio: number
  caracteristicas: Array<{
    caracteristica: string
    importancia: number
  }>
}

interface ValidacionCruzada {
  scores_por_fold: Array<{
    fold: number
    random_forest_r2: number
    linear_regression_r2: number
    random_forest_mse: number
    linear_regression_mse: number
  }>
  random_forest: {
    r2_promedio: number
    r2_desviacion: number
    mse_promedio: number
    mse_desviacion: number
  }
  linear_regression: {
    r2_promedio: number
    r2_desviacion: number
    mse_promedio: number
    mse_desviacion: number
  }
}

interface AnalisisResiduales {
  metricas: {
    mse: number
    rmse: number
    r2: number
    residual_promedio: number
    residual_std: number
  }
  qq_plot: Array<{ teorico: number; observado: number }>
  scatter_residuales: Array<{ predicho: number; residual: number }>
  histograma_residuales: Array<{ rango: string; frecuencia: number }>
  outliers: Array<{
    indice: number
    residual: number
    residual_estandarizado: number
    valor_real: number
    valor_predicho: number
  }>
  total_outliers: number
}

interface Multicolinealidad {
  vif: Array<{
    variable: string
    vif: number
    nivel: string
    color: string
  }>
  pares_alta_correlacion: Array<{
    variable1: string
    variable2: string
    correlacion: number
    abs_correlacion: number
  }>
  matriz_correlacion: Array<{ x: string; y: string; value: number }>
  variables: string[]
  resumen: {
    total_variables: number
    variables_alto_vif: number
    pares_correlacionados: number
  }
}

export default function AnalisisAvanzado() {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const datasetId = searchParams.get('dataset')
  
  const [datosRadar, setDatosRadar] = useState<RadarData[]>([])
  const [datosSankey, setDatosSankey] = useState<SankeyData>({ nodes: [], links: [] })
  const [datosBurbujas, setDatosBurbujas] = useState<BurbujaData[]>([])
  const [datosSensibilidad, setDatosSensibilidad] = useState<SensibilidadData[]>([])
  const [datosClustering, setDatosClustering] = useState<ClusterData>({ clusters: [], centroides: [], columnas: [] })
  const [rankingZonas, setRankingZonas] = useState<RankingZona[]>([])
  const [validacionCruzada, setValidacionCruzada] = useState<ValidacionCruzada | null>(null)
  const [analisisResiduales, setAnalisisResiduales] = useState<AnalisisResiduales | null>(null)
  const [multicolinealidad, setMulticolinealidad] = useState<Multicolinealidad | null>(null)
  const [cargando, setCargando] = useState(true)
  const [vistaActiva, setVistaActiva] = useState<'radar' | 'sankey' | 'burbujas' | 'sensibilidad' | 'clustering' | 'ranking' | 'validacion' | 'residuales' | 'multicolinealidad'>('radar')

  const COLORES = ['#0ea5e9', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#ef4444']

  useEffect(() => {
    if (datasetId) {
      cargarDatos()
    }
  }, [datasetId])

  const cargarDatos = async () => {
    setCargando(true)
    try {
      const [radar, sankey, burbujas, sensibilidad, clustering, ranking, validacion, residuales, multicol] = await Promise.all([
        axios.get(`http://localhost:5000/api/datasets/${datasetId}/analisis-radar`).catch(() => ({ data: [] })),
        axios.get(`http://localhost:5000/api/datasets/${datasetId}/sankey-flujo`).catch(() => ({ data: { nodes: [], links: [] } })),
        axios.get(`http://localhost:5000/api/datasets/${datasetId}/grafico-burbujas`).catch(() => ({ data: [] })),
        axios.get(`http://localhost:5000/api/datasets/${datasetId}/analisis-sensibilidad`).catch(() => ({ data: [] })),
        axios.get(`http://localhost:5000/api/datasets/${datasetId}/clustering`).catch(() => ({ data: { clusters: [], centroides: [], columnas: [] } })),
        axios.get(`http://localhost:5000/api/datasets/${datasetId}/ranking-caracteristicas-zona`).catch(() => ({ data: [] })),
        axios.get(`http://localhost:5000/api/datasets/${datasetId}/validacion-cruzada`).catch(() => ({ data: null })),
        axios.get(`http://localhost:5000/api/datasets/${datasetId}/analisis-residuales`).catch(() => ({ data: null })),
        axios.get(`http://localhost:5000/api/datasets/${datasetId}/multicolinealidad`).catch(() => ({ data: null }))
      ])

      setDatosRadar(Array.isArray(radar.data) ? radar.data : [])
      setDatosSankey(sankey.data || { nodes: [], links: [] })
      setDatosBurbujas(Array.isArray(burbujas.data) ? burbujas.data : [])
      setDatosSensibilidad(Array.isArray(sensibilidad.data) ? sensibilidad.data : [])
      setDatosClustering(clustering.data || { clusters: [], centroides: [], columnas: [] })
      setRankingZonas(Array.isArray(ranking.data) ? ranking.data : [])
      setValidacionCruzada(validacion.data)
      setAnalisisResiduales(residuales.data)
      setMulticolinealidad(multicol.data)
    } catch (error) {
      console.error('Error al cargar datos de análisis:', error)
    } finally {
      setCargando(false)
    }
  }

  const prepararDatosRadarChart = () => {
    if (datosRadar.length === 0) return []
    
    const metricas = datosRadar[0].metricas.map(m => m.metrica)
    const datos = metricas.map((metrica, idx) => {
      const punto: any = { metrica }
      datosRadar.forEach(casa => {
        punto[casa.casa] = casa.metricas[idx].valor
      })
      return punto
    })
    
    return datos
  }

  const prepararDatosSankeyVisualizacion = () => {
    if (datosSankey.nodes.length === 0) return []
    
    const datos: any[] = []
    
    datosSankey.links.forEach(link => {
      const sourceNode = datosSankey.nodes.find(n => n.id === link.source)
      const targetNode = datosSankey.nodes.find(n => n.id === link.target)
      
      if (sourceNode && targetNode) {
        datos.push({
          source: sourceNode.name,
          target: targetNode.name,
          value: link.value
        })
      }
    })
    
    return datos
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
          <p className="text-slate-600 text-lg mb-6">Para ver análisis avanzados</p>
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
          <h1 className="text-4xl md:text-5xl font-bold text-slate-900 mb-2">Análisis Avanzado</h1>
          <p className="text-slate-600 text-lg">Visualizaciones y análisis profundo de datos</p>
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
          onClick={() => setVistaActiva('radar')}
          className={`px-4 py-2 rounded-xl font-semibold transition-all whitespace-nowrap flex items-center space-x-2 ${
            vistaActiva === 'radar'
              ? 'bg-primary-600 text-white shadow-lg'
              : 'bg-white text-slate-700 border-2 border-slate-200'
          }`}
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z" />
          </svg>
          <span>Radar Comparativo</span>
        </button>

        <button
          onClick={() => setVistaActiva('sankey')}
          className={`px-4 py-2 rounded-xl font-semibold transition-all whitespace-nowrap flex items-center space-x-2 ${
            vistaActiva === 'sankey'
              ? 'bg-primary-600 text-white shadow-lg'
              : 'bg-white text-slate-700 border-2 border-slate-200'
          }`}
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4" />
          </svg>
          <span>Flujo Sankey</span>
        </button>

        <button
          onClick={() => setVistaActiva('burbujas')}
          className={`px-4 py-2 rounded-xl font-semibold transition-all whitespace-nowrap flex items-center space-x-2 ${
            vistaActiva === 'burbujas'
              ? 'bg-primary-600 text-white shadow-lg'
              : 'bg-white text-slate-700 border-2 border-slate-200'
          }`}
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
          </svg>
          <span>Gráfico Burbujas</span>
        </button>

        <button
          onClick={() => setVistaActiva('sensibilidad')}
          className={`px-4 py-2 rounded-xl font-semibold transition-all whitespace-nowrap flex items-center space-x-2 ${
            vistaActiva === 'sensibilidad'
              ? 'bg-primary-600 text-white shadow-lg'
              : 'bg-white text-slate-700 border-2 border-slate-200'
          }`}
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
          </svg>
          <span>Sensibilidad</span>
        </button>

        <button
          onClick={() => setVistaActiva('clustering')}
          className={`px-4 py-2 rounded-xl font-semibold transition-all whitespace-nowrap flex items-center space-x-2 ${
            vistaActiva === 'clustering'
              ? 'bg-primary-600 text-white shadow-lg'
              : 'bg-white text-slate-700 border-2 border-slate-200'
          }`}
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
          </svg>
          <span>Agrupamiento</span>
        </button>

        <button
          onClick={() => setVistaActiva('ranking')}
          className={`px-4 py-2 rounded-xl font-semibold transition-all whitespace-nowrap flex items-center space-x-2 ${
            vistaActiva === 'ranking'
              ? 'bg-primary-600 text-white shadow-lg'
              : 'bg-white text-slate-700 border-2 border-slate-200'
          }`}
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          <span>Ranking por Zona</span>
        </button>

        <button
          onClick={() => setVistaActiva('validacion')}
          className={`px-4 py-2 rounded-xl font-semibold transition-all whitespace-nowrap flex items-center space-x-2 ${
            vistaActiva === 'validacion'
              ? 'bg-primary-600 text-white shadow-lg'
              : 'bg-white text-slate-700 border-2 border-slate-200'
          }`}
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span>Validación Cruzada</span>
        </button>

        <button
          onClick={() => setVistaActiva('residuales')}
          className={`px-4 py-2 rounded-xl font-semibold transition-all whitespace-nowrap flex items-center space-x-2 ${
            vistaActiva === 'residuales'
              ? 'bg-primary-600 text-white shadow-lg'
              : 'bg-white text-slate-700 border-2 border-slate-200'
          }`}
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
          </svg>
          <span>Residuales</span>
        </button>

        <button
          onClick={() => setVistaActiva('multicolinealidad')}
          className={`px-4 py-2 rounded-xl font-semibold transition-all whitespace-nowrap flex items-center space-x-2 ${
            vistaActiva === 'multicolinealidad'
              ? 'bg-primary-600 text-white shadow-lg'
              : 'bg-white text-slate-700 border-2 border-slate-200'
          }`}
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
          </svg>
          <span>Multicolinealidad</span>
        </button>
      </div>

      {vistaActiva === 'radar' && (
        <div className="card p-8">
          <div className="flex items-center space-x-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center">
              <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z" />
              </svg>
            </div>
            <h2 className="text-3xl font-bold text-slate-900">Comparación Radar de Casas</h2>
          </div>
          {datosRadar.length > 0 ? (
            <ResponsiveContainer width="100%" height={500}>
              <RadarChart data={prepararDatosRadarChart()}>
                <PolarGrid stroke="#e2e8f0" />
                <PolarAngleAxis dataKey="metrica" stroke="#64748b" />
                <PolarRadiusAxis stroke="#64748b" domain={[0, 100]} />
                <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '2px solid #e2e8f0', borderRadius: '12px' }} />
                <Legend />
                {datosRadar.map((casa, idx) => (
                  <Radar key={casa.casa} name={casa.casa} dataKey={casa.casa} stroke={COLORES[idx % COLORES.length]} fill={COLORES[idx % COLORES.length]} fillOpacity={0.3} />
                ))}
              </RadarChart>
            </ResponsiveContainer>
          ) : (
            <div className="text-center py-12 text-slate-600">No hay datos disponibles para el análisis radar</div>
          )}
        </div>
      )}

      {vistaActiva === 'sankey' && (
        <div className="card p-8">
          <div className="flex items-center space-x-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
              <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4" />
              </svg>
            </div>
            <h2 className="text-3xl font-bold text-slate-900">Diagrama de Flujo Sankey</h2>
          </div>
          {prepararDatosSankeyVisualizacion().length > 0 ? (
            <div className="space-y-4">
              {prepararDatosSankeyVisualizacion().map((flujo, idx) => (
                <div key={idx} className="flex items-center gap-4 p-4 bg-slate-50 rounded-xl">
                  <div className="flex-1 text-sm font-semibold text-slate-700">{flujo.source}</div>
                  <div className="flex-1">
                    <div className="h-8 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg flex items-center justify-center text-white font-bold" style={{ width: `${Math.min(flujo.value * 10, 100)}%` }}>
                      {flujo.value}
                    </div>
                  </div>
                  <div className="flex-1 text-sm font-semibold text-slate-700 text-right">{flujo.target}</div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-12 text-slate-600">No hay datos disponibles para el diagrama Sankey</div>
          )}
        </div>
      )}

      {vistaActiva === 'burbujas' && (
        <div className="card p-8">
          <div className="flex items-center space-x-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-500 flex items-center justify-center">
              <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
              </svg>
            </div>
            <h2 className="text-3xl font-bold text-slate-900">Gráfico de Burbujas - Área vs Precio vs Zona</h2>
          </div>
          {datosBurbujas.length > 0 ? (
            <ResponsiveContainer width="100%" height={500}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="x" name="Área m²" stroke="#64748b" label={{ value: 'Área (m²)', position: 'insideBottom', offset: -5 }} />
                <YAxis dataKey="y" name="Precio" stroke="#64748b" label={{ value: 'Precio', angle: -90, position: 'insideLeft' }} />
                <ZAxis dataKey="z" range={[100, 1000]} name="Zona" />
                <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '2px solid #e2e8f0', borderRadius: '12px' }} cursor={{ strokeDasharray: '3 3' }} />
                <Legend />
                <Scatter name="Propiedades" data={datosBurbujas} fill="#8b5cf6">
                  {datosBurbujas.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORES[entry.z % COLORES.length]} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          ) : (
            <div className="text-center py-12 text-slate-600">No hay datos disponibles para el gráfico de burbujas</div>
          )}
        </div>
      )}

      {vistaActiva === 'sensibilidad' && (
        <div className="card p-8">
          <div className="flex items-center space-x-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-orange-500 to-red-500 flex items-center justify-center">
              <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
              </svg>
            </div>
            <h2 className="text-3xl font-bold text-slate-900">Análisis de Sensibilidad de Variables</h2>
          </div>
          {datosSensibilidad.length > 0 ? (
            <div className="space-y-6">
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={datosSensibilidad} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis type="number" stroke="#64748b" domain={[0, 100]} />
                  <YAxis dataKey="variable" type="category" stroke="#64748b" width={150} />
                  <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '2px solid #e2e8f0', borderRadius: '12px' }} />
                  <Legend />
                  <Bar dataKey="impacto_precio" fill="#f97316" name="Impacto en Precio (%)" radius={[0, 8, 8, 0]} />
                  <Bar dataKey="volatilidad" fill="#8b5cf6" name="Volatilidad (%)" radius={[0, 8, 8, 0]} />
                </BarChart>
              </ResponsiveContainer>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {datosSensibilidad.map((item, idx) => (
                  <div key={idx} className="p-5 bg-gradient-to-br from-slate-50 to-slate-100 rounded-xl border-2 border-slate-200">
                    <h3 className="font-bold text-slate-900 text-lg mb-3">{item.variable}</h3>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-slate-600">Impacto en Precio:</span>
                        <span className="text-lg font-bold text-orange-600">{item.impacto_precio}%</span>
                      </div>
                      <div className="w-full bg-slate-200 rounded-full h-2">
                        <div className="bg-orange-600 h-2 rounded-full transition-all duration-500" style={{ width: `${item.impacto_precio}%` }}></div>
                      </div>
                      <div className="flex justify-between items-center mt-3">
                        <span className="text-sm text-slate-600">Volatilidad:</span>
                        <span className="text-lg font-bold text-purple-600">{item.volatilidad}%</span>
                      </div>
                      <div className="w-full bg-slate-200 rounded-full h-2">
                        <div className="bg-purple-600 h-2 rounded-full transition-all duration-500" style={{ width: `${item.volatilidad}%` }}></div>
                      </div>
                      <div className="flex justify-between items-center mt-3 pt-3 border-t border-slate-300">
                        <span className="text-sm text-slate-600">Correlación:</span>
                        <span className="text-sm font-bold text-slate-900">{item.correlacion}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="text-center py-12 text-slate-600">No hay datos disponibles para el análisis de sensibilidad</div>
          )}
        </div>
      )}

      {vistaActiva === 'clustering' && (
        <div className="space-y-6">
          <div className="card p-8">
            <div className="flex items-center space-x-3 mb-6">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-pink-500 to-rose-500 flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                </svg>
              </div>
              <h2 className="text-3xl font-bold text-slate-900">Agrupamiento Automático de Casas Similares</h2>
            </div>
            {datosClustering.clusters.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {datosClustering.clusters.map((cluster) => (
                  <div key={cluster.id} className="p-6 bg-gradient-to-br from-white to-slate-50 rounded-2xl border-2 shadow-lg" style={{ borderColor: cluster.color }}>
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-2xl font-bold" style={{ color: cluster.color }}>{cluster.nombre}</h3>
                      <div className="px-4 py-2 rounded-full text-white font-bold" style={{ backgroundColor: cluster.color }}>
                        {cluster.cantidad} casas
                      </div>
                    </div>
                    <div className="space-y-4">
                      {Object.entries(cluster.caracteristicas).map(([variable, stats]: [string, any]) => (
                        <div key={variable} className="p-4 bg-white rounded-xl border border-slate-200">
                          <div className="font-semibold text-slate-900 mb-2 capitalize">{variable.replace('_', ' ')}</div>
                          <div className="grid grid-cols-3 gap-2 text-sm">
                            <div>
                              <span className="text-slate-600">Promedio:</span>
                              <div className="font-bold text-slate-900">{stats.promedio}</div>
                            </div>
                            <div>
                              <span className="text-slate-600">Mínimo:</span>
                              <div className="font-bold text-slate-900">{stats.min}</div>
                            </div>
                            <div>
                              <span className="text-slate-600">Máximo:</span>
                              <div className="font-bold text-slate-900">{stats.max}</div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-12 text-slate-600">No hay suficientes datos para realizar clustering</div>
            )}
          </div>

          {datosClustering.centroides.length > 0 && (
            <div className="card p-8">
              <h3 className="text-2xl font-bold text-slate-900 mb-6">Centroides de los Clusters</h3>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={datosClustering.centroides}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="cluster" stroke="#64748b" />
                  <YAxis stroke="#64748b" />
                  <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '2px solid #e2e8f0', borderRadius: '12px' }} />
                  <Legend />
                  {datosClustering.columnas.map((col, idx) => (
                    <Bar key={col} dataKey={col} fill={COLORES[idx % COLORES.length]} radius={[8, 8, 0, 0]} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      {vistaActiva === 'ranking' && (
        <div className="space-y-6">
          <div className="card p-8">
            <div className="flex items-center space-x-3 mb-6">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-indigo-500 to-blue-500 flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <h2 className="text-3xl font-bold text-slate-900">Ranking de Características por Zona</h2>
            </div>
            {rankingZonas.length > 0 ? (
              <div className="space-y-6">
                {rankingZonas.map((zona, idx) => (
                  <div key={zona.zona_id} className="p-6 bg-gradient-to-br from-white to-slate-50 rounded-2xl border-2 border-slate-200 animate-scale-in" style={{ animationDelay: `${idx * 0.1}s` }}>
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center space-x-3">
                        <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-indigo-500 to-blue-500 flex items-center justify-center text-white font-bold text-xl">
                          {zona.zona_id}
                        </div>
                        <div>
                          <h3 className="text-2xl font-bold text-slate-900">{zona.zona}</h3>
                          <p className="text-sm text-slate-600">{zona.total_propiedades} propiedades</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm text-slate-600">Precio Promedio</div>
                        <div className="text-2xl font-bold text-primary-600">${zona.precio_promedio.toLocaleString()}</div>
                      </div>
                    </div>
                    <div className="space-y-3">
                      {zona.caracteristicas.map((car, carIdx) => (
                        <div key={carIdx} className="p-3 bg-white rounded-xl border border-slate-200">
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center space-x-2">
                              <div className="w-6 h-6 rounded-full bg-gradient-to-br from-indigo-500 to-blue-500 flex items-center justify-center text-white text-xs font-bold">
                                {carIdx + 1}
                              </div>
                              <span className="font-semibold text-slate-900">{car.caracteristica}</span>
                            </div>
                            <span className="text-lg font-bold text-indigo-600">{car.importancia}%</span>
                          </div>
                          <div className="w-full bg-slate-200 rounded-full h-2">
                            <div className="bg-gradient-to-r from-indigo-500 to-blue-500 h-2 rounded-full transition-all duration-500" style={{ width: `${car.importancia}%` }}></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-12 text-slate-600">No hay datos disponibles para el ranking por zona</div>
            )}
          </div>
        </div>
      )}

      {vistaActiva === 'validacion' && (
        <div className="space-y-6">
          {validacionCruzada ? (
            <>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="card p-6 bg-gradient-to-br from-blue-50 to-indigo-50 border-2 border-blue-200">
                  <h3 className="text-xl font-bold text-slate-900 mb-4">Random Forest</h3>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center p-3 bg-white rounded-xl">
                      <span className="text-sm font-semibold text-slate-700">R² Promedio</span>
                      <span className="text-2xl font-bold text-blue-600">{validacionCruzada.random_forest.r2_promedio}</span>
                    </div>
                    <div className="flex justify-between items-center p-3 bg-white rounded-xl">
                      <span className="text-sm font-semibold text-slate-700">Desviación R²</span>
                      <span className="text-xl font-bold text-slate-900">{validacionCruzada.random_forest.r2_desviacion}</span>
                    </div>
                    <div className="flex justify-between items-center p-3 bg-white rounded-xl">
                      <span className="text-sm font-semibold text-slate-700">MSE Promedio</span>
                      <span className="text-xl font-bold text-slate-900">{validacionCruzada.random_forest.mse_promedio.toLocaleString()}</span>
                    </div>
                  </div>
                </div>

                <div className="card p-6 bg-gradient-to-br from-purple-50 to-pink-50 border-2 border-purple-200">
                  <h3 className="text-xl font-bold text-slate-900 mb-4">Regresión Lineal</h3>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center p-3 bg-white rounded-xl">
                      <span className="text-sm font-semibold text-slate-700">R² Promedio</span>
                      <span className="text-2xl font-bold text-purple-600">{validacionCruzada.linear_regression.r2_promedio}</span>
                    </div>
                    <div className="flex justify-between items-center p-3 bg-white rounded-xl">
                      <span className="text-sm font-semibold text-slate-700">Desviación R²</span>
                      <span className="text-xl font-bold text-slate-900">{validacionCruzada.linear_regression.r2_desviacion}</span>
                    </div>
                    <div className="flex justify-between items-center p-3 bg-white rounded-xl">
                      <span className="text-sm font-semibold text-slate-700">MSE Promedio</span>
                      <span className="text-xl font-bold text-slate-900">{validacionCruzada.linear_regression.mse_promedio.toLocaleString()}</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="card p-8">
                <h3 className="text-2xl font-bold text-slate-900 mb-6">Resultados por Fold</h3>
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart data={validacionCruzada.scores_por_fold}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="fold" stroke="#64748b" />
                    <YAxis stroke="#64748b" domain={[0, 1]} />
                    <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '2px solid #e2e8f0', borderRadius: '12px' }} />
                    <Legend />
                    <Line type="monotone" dataKey="random_forest_r2" stroke="#0ea5e9" strokeWidth={3} name="Random Forest R²" />
                    <Line type="monotone" dataKey="linear_regression_r2" stroke="#8b5cf6" strokeWidth={3} name="Regresión Lineal R²" />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div className="card p-8">
                <h3 className="text-2xl font-bold text-slate-900 mb-6">MSE por Fold</h3>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={validacionCruzada.scores_por_fold}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="fold" stroke="#64748b" />
                    <YAxis stroke="#64748b" />
                    <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '2px solid #e2e8f0', borderRadius: '12px' }} />
                    <Legend />
                    <Bar dataKey="random_forest_mse" fill="#0ea5e9" name="Random Forest MSE" radius={[8, 8, 0, 0]} />
                    <Bar dataKey="linear_regression_mse" fill="#8b5cf6" name="Regresión Lineal MSE" radius={[8, 8, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </>
          ) : (
            <div className="card p-16 text-center">
              <div className="text-slate-600">No hay datos de validación cruzada disponibles</div>
            </div>
          )}
        </div>
      )}

      {vistaActiva === 'residuales' && (
        <div className="space-y-6">
          {analisisResiduales ? (
            <>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="card p-6 bg-gradient-to-br from-blue-50 to-cyan-50 border-2 border-blue-200">
                  <div className="text-sm font-semibold text-blue-700 uppercase mb-2">MSE</div>
                  <div className="text-3xl font-bold text-blue-900">{analisisResiduales.metricas.mse.toLocaleString()}</div>
                </div>
                <div className="card p-6 bg-gradient-to-br from-purple-50 to-pink-50 border-2 border-purple-200">
                  <div className="text-sm font-semibold text-purple-700 uppercase mb-2">RMSE</div>
                  <div className="text-3xl font-bold text-purple-900">{analisisResiduales.metricas.rmse.toLocaleString()}</div>
                </div>
                <div className="card p-6 bg-gradient-to-br from-emerald-50 to-teal-50 border-2 border-emerald-200">
                  <div className="text-sm font-semibold text-emerald-700 uppercase mb-2">R²</div>
                  <div className="text-3xl font-bold text-emerald-900">{analisisResiduales.metricas.r2}</div>
                </div>
                <div className="card p-6 bg-gradient-to-br from-amber-50 to-orange-50 border-2 border-amber-200">
                  <div className="text-sm font-semibold text-amber-700 uppercase mb-2">Outliers</div>
                  <div className="text-3xl font-bold text-amber-900">{analisisResiduales.total_outliers}</div>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="card p-8">
                  <h3 className="text-2xl font-bold text-slate-900 mb-6">Q-Q Plot</h3>
                  <ResponsiveContainer width="100%" height={400}>
                    <ScatterChart>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="teorico" stroke="#64748b" name="Cuantiles Teóricos" />
                      <YAxis dataKey="observado" stroke="#64748b" name="Cuantiles Observados" />
                      <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '2px solid #e2e8f0', borderRadius: '12px' }} />
                      <Scatter data={analisisResiduales.qq_plot} fill="#8b5cf6" />
                      <Line type="linear" dataKey="teorico" stroke="#ef4444" strokeWidth={2} dot={false} />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>

                <div className="card p-8">
                  <h3 className="text-2xl font-bold text-slate-900 mb-6">Residuales vs Predichos</h3>
                  <ResponsiveContainer width="100%" height={400}>
                    <ScatterChart>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="predicho" stroke="#64748b" name="Valor Predicho" />
                      <YAxis dataKey="residual" stroke="#64748b" name="Residual" />
                      <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '2px solid #e2e8f0', borderRadius: '12px' }} />
                      <Scatter data={analisisResiduales.scatter_residuales} fill="#0ea5e9" />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>

                <div className="card p-8">
                  <h3 className="text-2xl font-bold text-slate-900 mb-6">Histograma de Residuales</h3>
                  <ResponsiveContainer width="100%" height={400}>
                    <BarChart data={analisisResiduales.histograma_residuales}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="rango" stroke="#64748b" />
                      <YAxis stroke="#64748b" />
                      <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '2px solid #e2e8f0', borderRadius: '12px' }} />
                      <Bar dataKey="frecuencia" fill="#10b981" radius={[8, 8, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <div className="card p-8">
                  <h3 className="text-2xl font-bold text-slate-900 mb-6">Outliers Detectados</h3>
                  <div className="space-y-3 max-h-96 overflow-y-auto">
                    {analisisResiduales.outliers.map((outlier, idx) => (
                      <div key={idx} className="p-4 bg-gradient-to-r from-rose-50 to-pink-50 rounded-xl border-2 border-rose-200">
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div>
                            <span className="text-slate-600">Índice:</span>
                            <div className="font-bold text-slate-900">{outlier.indice}</div>
                          </div>
                          <div>
                            <span className="text-slate-600">Residual:</span>
                            <div className="font-bold text-rose-600">{outlier.residual.toLocaleString()}</div>
                          </div>
                          <div>
                            <span className="text-slate-600">Real:</span>
                            <div className="font-bold text-slate-900">{outlier.valor_real.toLocaleString()}</div>
                          </div>
                          <div>
                            <span className="text-slate-600">Predicho:</span>
                            <div className="font-bold text-slate-900">{outlier.valor_predicho.toLocaleString()}</div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div className="card p-16 text-center">
              <div className="text-slate-600">No hay datos de análisis residual disponibles</div>
            </div>
          )}
        </div>
      )}

      {vistaActiva === 'multicolinealidad' && (
        <div className="space-y-6">
          {multicolinealidad ? (
            <>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="card p-6 bg-gradient-to-br from-blue-50 to-indigo-50 border-2 border-blue-200">
                  <div className="text-sm font-semibold text-blue-700 uppercase mb-2">Total Variables</div>
                  <div className="text-3xl font-bold text-blue-900">{multicolinealidad.resumen.total_variables}</div>
                </div>
                <div className="card p-6 bg-gradient-to-br from-rose-50 to-pink-50 border-2 border-rose-200">
                  <div className="text-sm font-semibold text-rose-700 uppercase mb-2">Alto VIF</div>
                  <div className="text-3xl font-bold text-rose-900">{multicolinealidad.resumen.variables_alto_vif}</div>
                </div>
                <div className="card p-6 bg-gradient-to-br from-amber-50 to-orange-50 border-2 border-amber-200">
                  <div className="text-sm font-semibold text-amber-700 uppercase mb-2">Pares Correlacionados</div>
                  <div className="text-3xl font-bold text-amber-900">{multicolinealidad.resumen.pares_correlacionados}</div>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="card p-8">
                  <h3 className="text-2xl font-bold text-slate-900 mb-6">Factor de Inflación de Varianza (VIF)</h3>
                  <div className="space-y-3 max-h-96 overflow-y-auto">
                    {multicolinealidad.vif.map((item, idx) => (
                      <div key={idx} className={`p-4 rounded-xl border-2 ${
                        item.color === 'danger' ? 'bg-rose-50 border-rose-200' :
                        item.color === 'warning' ? 'bg-amber-50 border-amber-200' :
                        'bg-emerald-50 border-emerald-200'
                      }`}>
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-semibold text-slate-900">{item.variable}</span>
                          <div className="flex items-center space-x-2">
                            <span className={`px-3 py-1 rounded-full text-xs font-bold ${
                              item.color === 'danger' ? 'bg-rose-600 text-white' :
                              item.color === 'warning' ? 'bg-amber-600 text-white' :
                              'bg-emerald-600 text-white'
                            }`}>
                              {item.nivel}
                            </span>
                            <span className="text-2xl font-bold text-slate-900">{item.vif}</span>
                          </div>
                        </div>
                        <div className={`w-full rounded-full h-2 ${
                          item.color === 'danger' ? 'bg-rose-200' :
                          item.color === 'warning' ? 'bg-amber-200' :
                          'bg-emerald-200'
                        }`}>
                          <div className={`h-2 rounded-full transition-all duration-500 ${
                            item.color === 'danger' ? 'bg-rose-600' :
                            item.color === 'warning' ? 'bg-amber-600' :
                            'bg-emerald-600'
                          }`} style={{ width: `${Math.min(item.vif * 5, 100)}%` }}></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="card p-8">
                  <h3 className="text-2xl font-bold text-slate-900 mb-6">Pares de Alta Correlación</h3>
                  <div className="space-y-3 max-h-96 overflow-y-auto">
                    {multicolinealidad.pares_alta_correlacion.map((par, idx) => (
                      <div key={idx} className="p-4 bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl border-2 border-purple-200">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center space-x-2">
                            <span className="font-semibold text-slate-900">{par.variable1}</span>
                            <svg className="w-4 h-4 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
                            </svg>
                            <span className="font-semibold text-slate-900">{par.variable2}</span>
                          </div>
                          <span className="text-xl font-bold text-purple-600">{par.correlacion}</span>
                        </div>
                        <div className="w-full bg-purple-200 rounded-full h-2">
                          <div className="bg-purple-600 h-2 rounded-full transition-all duration-500" style={{ width: `${par.abs_correlacion * 100}%` }}></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="card p-8">
                <h3 className="text-2xl font-bold text-slate-900 mb-6">Matriz de Correlación</h3>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr>
                        <th className="p-2 border border-slate-200"></th>
                        {multicolinealidad.variables.map((v, idx) => (
                          <th key={idx} className="p-2 border border-slate-200 text-xs font-semibold text-slate-700 transform -rotate-45 origin-bottom-left whitespace-nowrap" style={{ minWidth: '40px', maxWidth: '40px' }}>
                            {v}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {multicolinealidad.variables.map((v1, i) => (
                        <tr key={i}>
                          <td className="p-2 border border-slate-200 text-xs font-semibold text-slate-700 whitespace-nowrap">{v1}</td>
                          {multicolinealidad.variables.map((v2, j) => {
                            const valor = multicolinealidad.matriz_correlacion.find(m => m.x === v1 && m.y === v2)?.value || 0
                            const absValor = Math.abs(valor)
                            const color = absValor > 0.8 ? 'bg-red-500' : absValor > 0.6 ? 'bg-orange-500' : absValor > 0.4 ? 'bg-yellow-500' : absValor > 0.2 ? 'bg-green-500' : 'bg-blue-500'
                            const opacity = Math.round(absValor * 100)
                            return (
                              <td key={j} className="p-0 border border-slate-200 text-center relative group" style={{ minWidth: '40px', maxWidth: '40px', height: '40px' }}>
                                <div className={`absolute inset-0 ${color}`} style={{ opacity: opacity / 100 }}></div>
                                <div className="relative z-10 text-xs font-bold text-slate-900 opacity-0 group-hover:opacity-100 transition-opacity">
                                  {valor.toFixed(2)}
                                </div>
                              </td>
                            )
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="flex items-center justify-center space-x-4 mt-6">
                  <div className="flex items-center space-x-2">
                    <div className="w-8 h-8 bg-red-500 rounded"></div>
                    <span className="text-sm text-slate-600">0.8 - 1.0</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-8 h-8 bg-orange-500 rounded"></div>
                    <span className="text-sm text-slate-600">0.6 - 0.8</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-8 h-8 bg-yellow-500 rounded"></div>
                    <span className="text-sm text-slate-600">0.4 - 0.6</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-8 h-8 bg-green-500 rounded"></div>
                    <span className="text-sm text-slate-600">0.2 - 0.4</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-8 h-8 bg-blue-500 rounded"></div>
                    <span className="text-sm text-slate-600">0.0 - 0.2</span>
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div className="card p-16 text-center">
              <div className="text-slate-600">No hay datos de multicolinealidad disponibles</div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}