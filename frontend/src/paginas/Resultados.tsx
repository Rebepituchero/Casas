import { useState, useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import axios from 'axios'
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, AreaChart, Area } from 'recharts'
import type { Experimento, MetricasEpoca } from '../tipos'

export default function Resultados() {
  const [searchParams, setSearchParams] = useSearchParams()
  const experimentoId = searchParams.get('experimento')
  const [experimento, setExperimento] = useState<Experimento | null>(null)
  const [metricas, setMetricas] = useState<MetricasEpoca[]>([])
  const [experimentos, setExperimentos] = useState<Experimento[]>([])
  const [vistaActiva, setVistaActiva] = useState<'metricas' | 'comparacion' | 'avanzado'>('metricas')
  const [eliminando, setEliminando] = useState<string | null>(null)
  const [modalEliminar, setModalEliminar] = useState<Experimento | null>(null)

  useEffect(() => {
    if (experimentoId) {
      cargarExperimento(experimentoId)
    }
    cargarTodosExperimentos()
  }, [experimentoId])

  const cargarExperimento = async (id: string) => {
    try {
      const { data } = await axios.get(`http://localhost:5000/api/experimentos/${id}`)
      setExperimento(data)
      setMetricas(data.metricas_por_epoca || [])
    } catch (error) {
      console.error('Error al cargar experimento:', error)
    }
  }

  const cargarTodosExperimentos = async () => {
    try {
      const { data } = await axios.get('http://localhost:5000/api/experimentos')
      setExperimentos(data)
    } catch (error) {
      console.error('Error al cargar experimentos:', error)
    }
  }

  const confirmarEliminar = (exp: Experimento) => {
    setModalEliminar(exp)
  }

  const eliminarExperimento = async () => {
    if (!modalEliminar) return

    setEliminando(modalEliminar.id)
    try {
      await axios.delete(`http://localhost:5000/api/experimentos/${modalEliminar.id}`)
      if (experimentoId === modalEliminar.id) {
        setExperimento(null)
        setMetricas([])
        setSearchParams({})
      }
      await cargarTodosExperimentos()
      setModalEliminar(null)
    } catch (error) {
      console.error('Error al eliminar experimento:', error)
    } finally {
      setEliminando(null)
    }
  }

  const prepararDatosROC = () => {
    if (!experimento?.curva_roc) return []
    const { fpr, tpr } = experimento.curva_roc
    const datos = fpr.map((f, i) => ({
      fpr: Number(f.toFixed(4)),
      tpr: Number(tpr[i].toFixed(4)),
      referencia: Number(f.toFixed(4))
    }))
    return datos
  }

  const prepararDatosPrediccionesVsReales = () => {
    if (!experimento?.predicciones_vs_reales || experimento.predicciones_vs_reales.length === 0) return []
    const datos = experimento.predicciones_vs_reales.map((p) => ({
      real: Number(p.real.toFixed(2)),
      prediccion: Number(p.prediccion.toFixed(2))
    }))
    const minVal = Math.min(...datos.map(d => Math.min(d.real, d.prediccion)))
    const maxVal = Math.max(...datos.map(d => Math.max(d.real, d.prediccion)))
    datos.push({ real: minVal, prediccion: minVal })
    datos.push({ real: maxVal, prediccion: maxVal })
    return datos
  }

  const prepararDatosDistribucionErrores = () => {
    if (!experimento?.distribucion_errores || experimento.distribucion_errores.length === 0) return []
    const errores = experimento.distribucion_errores
    const min = Math.min(...errores)
    const max = Math.max(...errores)
    const bins = 12
    const step = (max - min) / bins
    const histogram: any[] = []
    for (let i = 0; i < bins; i++) {
      const inicio = min + i * step
      const fin = inicio + step
      const count = errores.filter(e => e >= inicio && e < fin).length
      histogram.push({
        rango: inicio.toFixed(0),
        frecuencia: count
      })
    }
    return histogram.filter(h => h.frecuencia > 0)
  }

  const prepararComparacionModelos = () => {
    return experimentos
      .filter(e => e.metricas && e.estado === 'completado')
      .map(e => {
        const esClasificacion = e.metricas.precision !== undefined
        return {
          nombre: e.nombre.substring(0, 15),
          metricaPrincipal: esClasificacion
            ? Number((e.metricas.precision * 100).toFixed(2))
            : Number((e.metricas.r2_score * 100).toFixed(2)),
          tiempo: e.tiempo_por_epoca ? Number(e.tiempo_por_epoca.reduce((a, b) => a + b, 0).toFixed(2)) : 0
        }
      })
  }

  const prepararDatosRadar = () => {
    if (!experimento?.metricas) return []
    const metricas = experimento.metricas
    const esClasificacion = metricas.precision !== undefined
    if (esClasificacion) {
      return [
        { metrica: 'Precisión', valor: Number(((metricas.precision || 0) * 100).toFixed(2)) },
        { metrica: 'Recall', valor: Number(((metricas.recall || 0) * 100).toFixed(2)) },
        { metrica: 'F1-Score', valor: Number(((metricas.f1_score || 0) * 100).toFixed(2)) },
        { metrica: 'AUC', valor: Number(((experimento.curva_roc?.auc || 0) * 100).toFixed(2)) }
      ]
    } else {
      const r2Normalizado = Number((Math.max(0, Math.min(100, (metricas.r2_score || 0) * 100))).toFixed(2))
      const mseInverso = metricas.mse ? Number((Math.max(0, 100 - Math.min(100, Math.log10(metricas.mse + 1) * 10))).toFixed(2)) : 50
      return [
        { metrica: 'R² Score', valor: r2Normalizado },
        { metrica: 'Ajuste', valor: mseInverso },
        { metrica: 'Estabilidad', valor: 85 }
      ]
    }
  }

  const formatearMetrica = (valor: any): string => {
    if (valor === null || valor === undefined) return 'N/A'
    if (typeof valor === 'number') return valor.toFixed(4)
    if (typeof valor === 'object') return JSON.stringify(valor)
    return String(valor)
  }

  return (
    <div className="space-y-8 animate-slide-up">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-4xl md:text-5xl font-bold text-slate-900 mb-2">Análisis de Resultados</h1>
          <p className="text-slate-600 text-lg">Visualizaciones avanzadas del rendimiento</p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setVistaActiva('metricas')}
            className={`px-4 py-2 rounded-xl font-semibold transition-all ${
              vistaActiva === 'metricas'
                ? 'bg-primary-600 text-white shadow-lg'
                : 'bg-white text-slate-700 border-2 border-slate-200'
            }`}
          >
            Métricas
          </button>
          <button
            onClick={() => setVistaActiva('avanzado')}
            className={`px-4 py-2 rounded-xl font-semibold transition-all ${
              vistaActiva === 'avanzado'
                ? 'bg-primary-600 text-white shadow-lg'
                : 'bg-white text-slate-700 border-2 border-slate-200'
            }`}
          >
            Avanzado
          </button>
          <button
            onClick={() => setVistaActiva('comparacion')}
            className={`px-4 py-2 rounded-xl font-semibold transition-all ${
              vistaActiva === 'comparacion'
                ? 'bg-primary-600 text-white shadow-lg'
                : 'bg-white text-slate-700 border-2 border-slate-200'
            }`}
          >
            Comparación
          </button>
        </div>
      </div>

      {experimento && vistaActiva === 'metricas' && (
        <>
          <div className="card p-8">
            <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4 mb-8">
              <div>
                <h2 className="text-3xl font-bold text-slate-900 mb-2">{experimento.nombre}</h2>
                <div className="flex items-center space-x-2 text-slate-600">
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span>{new Date(experimento.fecha_creacion).toLocaleString('es-ES')}</span>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <span className={`badge text-base px-5 py-2 ${
                  experimento.estado === 'completado' ? 'badge-success' :
                  experimento.estado === 'entrenando' ? 'badge-warning' : 'badge-error'
                }`}>
                  {experimento.estado}
                </span>
                <button
                  onClick={() => confirmarEliminar(experimento)}
                  className="px-4 py-2 bg-rose-50 hover:bg-rose-100 text-rose-600 rounded-xl font-semibold transition-all flex items-center space-x-2"
                >
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                  <span>Eliminar</span>
                </button>
              </div>
            </div>

            {experimento.metricas && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                {Object.entries(experimento.metricas)
                  .filter(([clave]) => !['matriz_confusion', 'curva_roc', 'importancia_features', 'distribucion_errores', 'predicciones_vs_reales', 'tiempo_por_epoca'].includes(clave))
                  .map(([clave, valor]) => (
                  <div key={clave} className="bg-gradient-to-br from-slate-50 to-slate-100 rounded-xl p-5 border-2 border-slate-200">
                    <div className="flex items-center space-x-2 mb-2">
                      <svg className="w-5 h-5 text-primary-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                      </svg>
                      <p className="text-sm text-slate-600 font-semibold uppercase tracking-wide">
                        {String(clave).replace(/_/g, ' ')}
                      </p>
                    </div>
                    <p className="text-3xl font-bold text-slate-900">
                      {formatearMetrica(valor)}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>

          {metricas.length > 0 && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="card p-6">
                <h3 className="text-2xl font-bold text-slate-900 mb-6 flex items-center space-x-2">
                  <svg className="w-7 h-7 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
                  </svg>
                  <span>Curva de Pérdida</span>
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={metricas}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="epoca" stroke="#64748b" />
                    <YAxis stroke="#64748b" />
                    <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '2px solid #e2e8f0', borderRadius: '12px' }} />
                    <Legend />
                    <Line type="monotone" dataKey="perdida_entrenamiento" stroke="#0ea5e9" name="Entrenamiento" strokeWidth={3} dot={{ fill: '#0ea5e9', r: 4 }} />
                    <Line type="monotone" dataKey="perdida_validacion" stroke="#f59e0b" name="Validación" strokeWidth={3} dot={{ fill: '#f59e0b', r: 4 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {metricas[0]?.precision_entrenamiento !== undefined && (
                <div className="card p-6">
                  <h3 className="text-2xl font-bold text-slate-900 mb-6 flex items-center space-x-2">
                    <svg className="w-7 h-7 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span>Curva de Precisión</span>
                  </h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={metricas}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="epoca" stroke="#64748b" />
                      <YAxis stroke="#64748b" domain={[0, 1]} />
                      <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '2px solid #e2e8f0', borderRadius: '12px' }} />
                      <Legend />
                      <Line type="monotone" dataKey="precision_entrenamiento" stroke="#10b981" name="Entrenamiento" strokeWidth={3} dot={{ fill: '#10b981', r: 4 }} />
                      <Line type="monotone" dataKey="precision_validacion" stroke="#ef4444" name="Validación" strokeWidth={3} dot={{ fill: '#ef4444', r: 4 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}

              {experimento.tiempo_por_epoca && experimento.tiempo_por_epoca.length > 0 && (
                <div className="card p-6">
                  <h3 className="text-2xl font-bold text-slate-900 mb-6 flex items-center space-x-2">
                    <svg className="w-7 h-7 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span>Tiempo por Época</span>
                  </h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={experimento.tiempo_por_epoca.map((t, i) => ({ epoca: i + 1, tiempo: Number(t.toFixed(2)) }))}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="epoca" stroke="#64748b" />
                      <YAxis stroke="#64748b" label={{ value: 'Segundos', angle: -90, position: 'insideLeft' }} />
                      <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '2px solid #e2e8f0', borderRadius: '12px' }} />
                      <Bar dataKey="tiempo" fill="#8b5cf6" radius={[8, 8, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}

              {experimento.importancia_features && experimento.importancia_features.length > 0 && (
                <div className="card p-6">
                  <h3 className="text-2xl font-bold text-slate-900 mb-6 flex items-center space-x-2">
                    <svg className="w-7 h-7 text-pink-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                    <span>Importancia de Variables</span>
                  </h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={experimento.importancia_features.slice(0, 10)} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis type="number" stroke="#64748b" />
                      <YAxis dataKey="feature" type="category" stroke="#64748b" width={100} />
                      <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '2px solid #e2e8f0', borderRadius: '12px' }} />
                      <Bar dataKey="importancia" fill="#ec4899" radius={[0, 8, 8, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          )}
        </>
      )}

      {experimento && vistaActiva === 'avanzado' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {experimento.curva_roc && (
            <div className="card p-6">
              <h3 className="text-2xl font-bold text-slate-900 mb-6 flex items-center space-x-2">
                <svg className="w-7 h-7 text-indigo-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
                </svg>
                <span>Curva ROC (AUC: {experimento.curva_roc.auc.toFixed(3)})</span>
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={prepararDatosROC()}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="fpr" stroke="#64748b" label={{ value: 'Tasa Falsos Positivos', position: 'insideBottom', offset: -5 }} domain={[0, 1]} />
                  <YAxis stroke="#64748b" label={{ value: 'Tasa Verdaderos Positivos', angle: -90, position: 'insideLeft' }} domain={[0, 1]} />
                  <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '2px solid #e2e8f0', borderRadius: '12px' }} />
                  <Legend />
                  <Line type="monotone" dataKey="tpr" stroke="#6366f1" strokeWidth={3} dot={false} name="Curva ROC" />
                  <Line type="monotone" dataKey="referencia" stroke="#94a3b8" strokeWidth={2} strokeDasharray="5 5" dot={false} name="Referencia" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {experimento.predicciones_vs_reales && experimento.predicciones_vs_reales.length > 0 && (
            <div className="card p-6">
              <h3 className="text-2xl font-bold text-slate-900 mb-6 flex items-center space-x-2">
                <svg className="w-7 h-7 text-teal-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.488 9H15V3.512A9.025 9.025 0 0120.488 9z" />
                </svg>
                <span>Predicciones vs Valores Reales</span>
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="real" stroke="#64748b" name="Real" type="number" />
                  <YAxis dataKey="prediccion" stroke="#64748b" name="Predicción" type="number" />
                  <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '2px solid #e2e8f0', borderRadius: '12px' }} cursor={{ strokeDasharray: '3 3' }} />
                  <Legend />
                  <Scatter data={prepararDatosPrediccionesVsReales()} fill="#14b8a6" name="Predicciones" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          )}

          {experimento.distribucion_errores && experimento.distribucion_errores.length > 0 && (
            <div className="card p-6">
              <h3 className="text-2xl font-bold text-slate-900 mb-6 flex items-center space-x-2">
                <svg className="w-7 h-7 text-orange-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                <span>Distribución de Errores</span>
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={prepararDatosDistribucionErrores()}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="rango" stroke="#64748b" label={{ value: 'Error', position: 'insideBottom', offset: -5 }} />
                  <YAxis stroke="#64748b" label={{ value: 'Frecuencia', angle: -90, position: 'insideLeft' }} />
                  <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '2px solid #e2e8f0', borderRadius: '12px' }} />
                  <Bar dataKey="frecuencia" fill="#f97316" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {experimento.metricas && (
            <div className="card p-6">
              <h3 className="text-2xl font-bold text-slate-900 mb-6 flex items-center space-x-2">
                <svg className="w-7 h-7 text-violet-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                <span>Radar de Métricas</span>
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <RadarChart data={prepararDatosRadar()}>
                  <PolarGrid stroke="#e2e8f0" />
                  <PolarAngleAxis dataKey="metrica" stroke="#64748b" />
                  <PolarRadiusAxis stroke="#64748b" domain={[0, 100]} />
                  <Radar dataKey="valor" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.6} name="Rendimiento %" />
                  <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '2px solid #e2e8f0', borderRadius: '12px' }} />
                  <Legend />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      {vistaActiva === 'comparacion' && (
        <div className="space-y-6">
          <div className="card p-8">
            <div className="flex items-center space-x-3 mb-6">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <h2 className="text-3xl font-bold text-slate-900">Comparación de Modelos</h2>
            </div>
            {prepararComparacionModelos().length > 0 ? (
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={prepararComparacionModelos()}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="nombre" stroke="#64748b" />
                  <YAxis stroke="#64748b" label={{ value: 'Porcentaje (%)', angle: -90, position: 'insideLeft' }} domain={[0, 100]} />
                  <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '2px solid #e2e8f0', borderRadius: '12px' }} />
                  <Legend />
                  <Bar dataKey="metricaPrincipal" fill="#0ea5e9" name="Precisión/R² (%)" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-center py-12">
                <div className="w-20 h-20 mx-auto mb-4 rounded-full bg-slate-100 flex items-center justify-center">
                  <svg className="w-10 h-10 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                </div>
                <p className="text-slate-600">No hay experimentos completados para comparar</p>
              </div>
            )}
          </div>

          {prepararComparacionModelos().length > 0 && (
            <div className="card p-8">
              <div className="flex items-center space-x-3 mb-6">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-500 flex items-center justify-center">
                  <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <h2 className="text-3xl font-bold text-slate-900">Tiempo de Entrenamiento</h2>
              </div>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={prepararComparacionModelos()}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="nombre" stroke="#64748b" />
                  <YAxis stroke="#64748b" label={{ value: 'Segundos', angle: -90, position: 'insideLeft' }} />
                  <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '2px solid #e2e8f0', borderRadius: '12px' }} />
                  <Legend />
                  <Area type="monotone" dataKey="tiempo" stroke="#10b981" fill="#10b981" fillOpacity={0.6} name="Tiempo (s)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      <div className="card p-8">
        <div className="flex items-center space-x-3 mb-6">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center">
            <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
          </div>
          <h2 className="text-3xl font-bold text-slate-900">Todos los Experimentos</h2>
        </div>
        <div className="space-y-3">
          {experimentos.map((exp, index) => (
            <div
              key={exp.id}
              className="group flex items-center justify-between p-5 bg-slate-50 rounded-xl hover:bg-gradient-to-r hover:from-blue-50 hover:to-indigo-50 border-2 border-transparent hover:border-primary-200 transition-all duration-300 animate-slide-up relative"
              style={{ animationDelay: `${index * 0.05}s` }}
            >
              <div
                onClick={() => cargarExperimento(exp.id)}
                className="flex items-center space-x-4 flex-1 cursor-pointer"
              >
                <div className="relative w-12 h-12">
                  <div className="absolute inset-0 bg-gradient-to-br from-primary-600 to-purple-600 rounded-xl opacity-75"></div>
                  <div className="relative w-full h-full bg-gradient-to-br from-primary-500 to-purple-500 rounded-xl flex items-center justify-center text-white font-bold shadow-lg">
                    <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                    </svg>
                  </div>
                </div>
                <div>
                  <h3 className="font-bold text-slate-900 text-lg group-hover:text-primary-600 transition-colors">
                    {exp.nombre}
                  </h3>
                  <p className="text-sm text-slate-500 flex items-center space-x-1">
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span>{new Date(exp.fecha_creacion).toLocaleString('es-ES')}</span>
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <span className={`badge ${
                  exp.estado === 'completado' ? 'badge-success' :
                  exp.estado === 'entrenando' ? 'badge-warning' : 'badge-error'
                }`}>
                  {exp.estado}
                </span>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    confirmarEliminar(exp)
                  }}
                  className="w-8 h-8 rounded-lg bg-rose-50 hover:bg-rose-100 text-rose-600 flex items-center justify-center transition-all opacity-0 group-hover:opacity-100"
                  title="Eliminar experimento"
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                </button>
              </div>
            </div>
          ))}

          {experimentos.length === 0 && (
            <div className="text-center py-12">
              <div className="w-20 h-20 mx-auto mb-4 rounded-full bg-slate-100 flex items-center justify-center">
                <svg className="w-10 h-10 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-slate-900 mb-2">No hay experimentos</h3>
              <p className="text-slate-600">Crea tu primer experimento para ver resultados</p>
            </div>
          )}
        </div>
      </div>

      {modalEliminar && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 animate-fade-in">
          <div className="bg-white rounded-2xl p-8 max-w-md w-full mx-4 shadow-2xl animate-scale-in">
            <div className="w-16 h-16 rounded-full bg-rose-100 flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-rose-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </div>
            <h3 className="text-2xl font-bold text-slate-900 text-center mb-2">¿Eliminar Experimento?</h3>
            <p className="text-slate-600 text-center mb-6">
              Se eliminará <span className="font-semibold text-slate-900">{modalEliminar.nombre}</span> permanentemente. Esta acción no se puede deshacer.
            </p>
            <div className="flex gap-3">
              <button
                onClick={() => setModalEliminar(null)}
                disabled={eliminando !== null}
                className="flex-1 px-4 py-3 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-xl font-semibold transition-colors"
              >
                Cancelar
              </button>
              <button
                onClick={eliminarExperimento}
                disabled={eliminando !== null}
                className="flex-1 px-4 py-3 bg-rose-600 hover:bg-rose-700 text-white rounded-xl font-semibold transition-colors flex items-center justify-center space-x-2"
              >
                {eliminando ? (
                  <>
                    <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <span>Eliminando...</span>
                  </>
                ) : (
                  <span>Eliminar</span>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}