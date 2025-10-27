// frontend/src/paginas/Limpieza.tsx

import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import axios from 'axios'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line, ScatterChart, Scatter } from 'recharts'
import type { ColumnaStat, EstadisticasDatos, EstadisticasLimpieza, ResultadoLimpieza } from '../tipos'

export default function Limpieza() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [columnas, setColumnas] = useState<ColumnaStat[]>([])
  const [vistaPrevia, setVistaPrevia] = useState<any[]>([])
  const [vistaPreviaFiltrada, setVistaPreviaFiltrada] = useState<any[]>([])
  const [procesando, setProcesando] = useState(false)
  const [distribucionClases, setDistribucionClases] = useState<any[]>([])
  const [estadisticasDatos, setEstadisticasDatos] = useState<EstadisticasDatos | null>(null)
  const [filtroBusqueda, setFiltroBusqueda] = useState('')
  const [estadisticasLimpieza, setEstadisticasLimpieza] = useState<EstadisticasLimpieza | null>(null)
  const [mostrarResultados, setMostrarResultados] = useState(false)
  const [datasetLimpioId, setDatasetLimpioId] = useState<string | null>(null)
  const [paginaActual, setPaginaActual] = useState(1)
  const [totalPaginas, setTotalPaginas] = useState(1)
  const [totalFilas, setTotalFilas] = useState(0)
  const [cargandoPagina, setCargandoPagina] = useState(false)
  const [mapaCalor, setMapaCalor] = useState<any[]>([])
  const [datos3D, setDatos3D] = useState<any[]>([])
  const [histogramaPrecio, setHistogramaPrecio] = useState<any[]>([])
  const [boxplotZonas, setBoxplotZonas] = useState<any[]>([])
  const [serieTemporal, setSerieTemporal] = useState<any[]>([])
  const [operaciones, setOperaciones] = useState({
    eliminar_nulos: true,
    normalizar: false,
    codificar_categoricas: true,
    detectar_outliers: false,
    eliminar_duplicados: true,
  })

  const COLORES = ['#0ea5e9', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#ef4444']

  useEffect(() => {
    cargarDatos()
  }, [id])

  useEffect(() => {
    if (!Array.isArray(vistaPrevia)) {
      setVistaPreviaFiltrada([])
      return
    }
    if (filtroBusqueda.trim() === '') {
      setVistaPreviaFiltrada(vistaPrevia)
    } else {
      const filtrado = vistaPrevia.filter(fila =>
        Object.values(fila).some(valor =>
          String(valor).toLowerCase().includes(filtroBusqueda.toLowerCase())
        )
      )
      setVistaPreviaFiltrada(filtrado)
    }
  }, [filtroBusqueda, vistaPrevia])

  const cargarDatos = async () => {
    try {
      const requests = [
        axios.get(`http://localhost:5000/api/datasets/${id}/columnas`).catch(() => ({ data: [] })),
        axios.get(`http://localhost:5000/api/datasets/${id}/vista-previa?pagina=1`).catch(() => ({ data: { datos: [], pagina_actual: 1, total_paginas: 1, total_filas: 0 } })),
        axios.get(`http://localhost:5000/api/datasets/${id}/distribucion-clases`).catch(() => ({ data: [] })),
        axios.get(`http://localhost:5000/api/datasets/${id}/estadisticas`).catch(() => ({ data: null })),
        axios.get(`http://localhost:5000/api/datasets/${id}/mapa-calor`).catch(() => ({ data: [] })),
        axios.get(`http://localhost:5000/api/datasets/${id}/datos-3d`).catch(() => ({ data: [] })),
        axios.get(`http://localhost:5000/api/datasets/${id}/histograma-precio`).catch(() => ({ data: [] })),
        axios.get(`http://localhost:5000/api/datasets/${id}/boxplot-zonas`).catch(() => ({ data: [] })),
        axios.get(`http://localhost:5000/api/datasets/${id}/serie-temporal`).catch(() => ({ data: [] }))
      ]

      const [respCol, respPrev, respDist, respEst, respMapaCalor, resp3D, respHist, respBox, respSerie] = await Promise.all(requests)

      setColumnas(respCol.data || [])
      const previewData = Array.isArray(respPrev.data.datos) ? respPrev.data.datos : []
      setVistaPrevia(previewData)
      setVistaPreviaFiltrada(previewData)
      setPaginaActual(respPrev.data.pagina_actual || 1)
      setTotalPaginas(respPrev.data.total_paginas || 1)
      setTotalFilas(respPrev.data.total_filas || 0)
      setDistribucionClases(respDist.data || [])
      setEstadisticasDatos(respEst.data || null)
      setMapaCalor(respMapaCalor.data || [])
      setDatos3D(resp3D.data || [])
      setHistogramaPrecio(respHist.data || [])
      setBoxplotZonas(respBox.data || [])
      setSerieTemporal(respSerie.data || [])
    } catch (error) {
      console.error('Error al cargar datos:', error)
      setVistaPrevia([])
      setVistaPreviaFiltrada([])
    }
  }

  const cargarPagina = async (numeroPagina: number) => {
    if (numeroPagina < 1 || numeroPagina > totalPaginas) return
    
    setCargandoPagina(true)
    try {
      const response = await axios.get(`http://localhost:5000/api/datasets/${id}/vista-previa?pagina=${numeroPagina}`)
      
      const previewData = Array.isArray(response.data.datos) ? response.data.datos : []
      setVistaPrevia(previewData)
      setVistaPreviaFiltrada(previewData)
      setPaginaActual(response.data.pagina_actual)
      setTotalPaginas(response.data.total_paginas)
      setTotalFilas(response.data.total_filas)
      
      const tableContainer = document.getElementById('vista-previa-table')
      if (tableContainer) {
        tableContainer.scrollTo(0, 0)
      }
    } catch (error) {
      console.error('Error al cargar página:', error)
    } finally {
      setCargandoPagina(false)
    }
  }

  const aplicarLimpieza = async () => {
    setProcesando(true)
    setMostrarResultados(false)
    try {
      const { data } = await axios.post<ResultadoLimpieza>(`http://localhost:5000/api/datasets/${id}/limpiar`, operaciones)
      setEstadisticasLimpieza(data.estadisticas)
      setDatasetLimpioId(data.dataset_limpio_id)
      setMostrarResultados(true)
    } catch (error) {
      console.error('Error al limpiar:', error)
      alert('Error al aplicar la limpieza')
    } finally {
      setProcesando(false)
    }
  }

  const descargarCSV = async () => {
    if (!datasetLimpioId) return
    try {
      const response = await axios.get(`http://localhost:5000/api/datasets/${datasetLimpioId}/descargar`, {
        responseType: 'blob'
      })
      const url = window.URL.createObjectURL(new Blob([response.data]))
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', `dataset_limpio_${Date.now()}.csv`)
      document.body.appendChild(link)
      link.click()
      link.remove()
    } catch (error) {
      console.error('Error al descargar:', error)
    }
  }

  const verDatasetLimpio = () => {
    if (datasetLimpioId) {
      navigate(`/limpieza/${datasetLimpioId}`)
      window.location.reload()
    }
  }

  const prepararDatosNulos = () => {
    return columnas.map(col => ({
      nombre: col.nombre.substring(0, 10),
      nulos: col.valores_nulos,
      completos: totalFilas - col.valores_nulos
    })).slice(0, 8)
  }

  return (
    <div className="space-y-8 animate-slide-up">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-4xl md:text-5xl font-bold text-slate-900 mb-2">Exploración y Limpieza</h1>
          <p className="text-slate-600 text-lg">Analiza y prepara tus datos</p>
        </div>
        <button onClick={() => navigate('/entrenamiento')} className="btn-primary space-x-2">
          <span>Continuar al Entrenamiento</span>
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
          </svg>
        </button>
      </div>

      {estadisticasDatos && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="card p-5">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-semibold text-slate-600 uppercase">Total Filas</span>
              <svg className="w-5 h-5 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 10h16M4 14h16M4 18h16" />
              </svg>
            </div>
            <p className="text-3xl font-bold text-slate-900">{estadisticasDatos.total_filas.toLocaleString()}</p>
          </div>
          <div className="card p-5">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-semibold text-slate-600 uppercase">Valores Nulos</span>
              <svg className="w-5 h-5 text-rose-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </div>
            <p className="text-3xl font-bold text-slate-900">{estadisticasDatos.total_nulos.toLocaleString()}</p>
            <p className="text-xs text-slate-500 mt-1">{estadisticasDatos.porcentaje_nulos.toFixed(2)}% del total</p>
          </div>
          <div className="card p-5">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-semibold text-slate-600 uppercase">Duplicados</span>
              <svg className="w-5 h-5 text-amber-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
            </div>
            <p className="text-3xl font-bold text-slate-900">{estadisticasDatos.total_duplicados.toLocaleString()}</p>
          </div>
          <div className="card p-5">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-semibold text-slate-600 uppercase">Columnas</span>
              <svg className="w-5 h-5 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 4v16m6-16v16m-9-9h12" />
              </svg>
            </div>
            <p className="text-3xl font-bold text-slate-900">{estadisticasDatos.total_columnas}</p>
          </div>
        </div>
      )}

      {mostrarResultados && estadisticasLimpieza && (
        <div className="card p-6 bg-gradient-to-r from-emerald-50 to-teal-50 border-2 border-emerald-200">
          <div className="flex items-start justify-between mb-4">
            <div className="flex items-center space-x-3">
              <div className="w-12 h-12 rounded-xl bg-emerald-500 flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div>
                <h3 className="text-2xl font-bold text-slate-900">Limpieza Completada</h3>
                <p className="text-slate-600">Resultados del proceso de limpieza</p>
              </div>
            </div>
            <button onClick={() => setMostrarResultados(false)} className="text-slate-400 hover:text-slate-600">
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-white rounded-xl p-4 border-2 border-emerald-200">
              <p className="text-sm text-slate-600 font-semibold mb-1">Filas Eliminadas</p>
              <p className="text-3xl font-bold text-emerald-600">{estadisticasLimpieza.filas_eliminadas}</p>
            </div>
            <div className="bg-white rounded-xl p-4 border-2 border-emerald-200">
              <p className="text-sm text-slate-600 font-semibold mb-1">Nulos Eliminados</p>
              <p className="text-3xl font-bold text-blue-600">{estadisticasLimpieza.nulos_eliminados}</p>
            </div>
            <div className="bg-white rounded-xl p-4 border-2 border-emerald-200">
              <p className="text-sm text-slate-600 font-semibold mb-1">Duplicados Eliminados</p>
              <p className="text-3xl font-bold text-amber-600">{estadisticasLimpieza.duplicados_eliminados}</p>
            </div>
            <div className="bg-white rounded-xl p-4 border-2 border-emerald-200">
              <p className="text-sm text-slate-600 font-semibold mb-1">Datos Conservados</p>
              <p className="text-3xl font-bold text-purple-600">{(100 - estadisticasLimpieza.porcentaje_datos_eliminados).toFixed(1)}%</p>
            </div>
          </div>
          <div className="flex gap-3">
            <button onClick={descargarCSV} className="btn-primary space-x-2">
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
              <span>Descargar CSV Limpio</span>
            </button>
            <button onClick={verDatasetLimpio} className="btn-secondary space-x-2">
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
              <span>Ver Dataset Limpio</span>
            </button>
            <button onClick={() => navigate('/datasets')} className="btn-secondary space-x-2">
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
              </svg>
              <span>Ir a Datasets</span>
            </button>
          </div>
        </div>
      )}

      <div className="card p-6">
        <div className="flex items-center space-x-3 mb-6">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-teal-500 flex items-center justify-center">
            <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </div>
          <h2 className="text-2xl font-bold text-slate-900">Operaciones de Limpieza</h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4 mb-6">
          <label className="flex items-start space-x-3 cursor-pointer p-4 bg-slate-50 rounded-xl hover:bg-slate-100 transition-colors group">
            <input type="checkbox" checked={operaciones.eliminar_nulos} onChange={(e) => setOperaciones({ ...operaciones, eliminar_nulos: e.target.checked })} className="w-5 h-5 text-primary-600 rounded focus:ring-primary-500 mt-0.5" />
            <div className="flex-1">
              <div className="font-semibold text-slate-900 group-hover:text-primary-600 transition-colors text-sm">Eliminar Valores Nulos</div>
              <div className="text-xs text-slate-600 mt-1">Remover filas con datos faltantes</div>
            </div>
          </label>
          <label className="flex items-start space-x-3 cursor-pointer p-4 bg-slate-50 rounded-xl hover:bg-slate-100 transition-colors group">
            <input type="checkbox" checked={operaciones.eliminar_duplicados} onChange={(e) => setOperaciones({ ...operaciones, eliminar_duplicados: e.target.checked })} className="w-5 h-5 text-primary-600 rounded focus:ring-primary-500 mt-0.5" />
            <div className="flex-1">
              <div className="font-semibold text-slate-900 group-hover:text-primary-600 transition-colors text-sm">Eliminar Duplicados</div>
              <div className="text-xs text-slate-600 mt-1">Remover filas duplicadas</div>
            </div>
          </label>
          <label className="flex items-start space-x-3 cursor-pointer p-4 bg-slate-50 rounded-xl hover:bg-slate-100 transition-colors group">
            <input type="checkbox" checked={operaciones.normalizar} onChange={(e) => setOperaciones({ ...operaciones, normalizar: e.target.checked })} className="w-5 h-5 text-primary-600 rounded focus:ring-primary-500 mt-0.5" />
            <div className="flex-1">
              <div className="font-semibold text-slate-900 group-hover:text-primary-600 transition-colors text-sm">Normalizar Datos</div>
              <div className="text-xs text-slate-600 mt-1">Escalar valores numéricos</div>
            </div>
          </label>
          <label className="flex items-start space-x-3 cursor-pointer p-4 bg-slate-50 rounded-xl hover:bg-slate-100 transition-colors group">
            <input type="checkbox" checked={operaciones.codificar_categoricas} onChange={(e) => setOperaciones({ ...operaciones, codificar_categoricas: e.target.checked })} className="w-5 h-5 text-primary-600 rounded focus:ring-primary-500 mt-0.5" />
            <div className="flex-1">
              <div className="font-semibold text-slate-900 group-hover:text-primary-600 transition-colors text-sm">Codificar Categóricas</div>
              <div className="text-xs text-slate-600 mt-1">Convertir texto a números</div>
            </div>
          </label>
          <label className="flex items-start space-x-3 cursor-pointer p-4 bg-slate-50 rounded-xl hover:bg-slate-100 transition-colors group">
            <input type="checkbox" checked={operaciones.detectar_outliers} onChange={(e) => setOperaciones({ ...operaciones, detectar_outliers: e.target.checked })} className="w-5 h-5 text-primary-600 rounded focus:ring-primary-500 mt-0.5" />
            <div className="flex-1">
              <div className="font-semibold text-slate-900 group-hover:text-primary-600 transition-colors text-sm">Detectar Outliers</div>
              <div className="text-xs text-slate-600 mt-1">Identificar valores atípicos</div>
            </div>
          </label>
        </div>
        <button onClick={aplicarLimpieza} disabled={procesando} className="w-full btn-primary">
          {procesando ? (
            <div className="flex items-center justify-center space-x-2">
              <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <span>Procesando...</span>
            </div>
          ) : (
            'Aplicar Limpieza'
          )}
        </button>
      </div>

      <div className="space-y-6">
        <div className="card p-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                </svg>
              </div>
              <h2 className="text-2xl font-bold text-slate-900">Vista Previa</h2>
            </div>
            <div className="flex items-center space-x-2">
              <svg className="w-5 h-5 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <input type="text" value={filtroBusqueda} onChange={(e) => setFiltroBusqueda(e.target.value)} placeholder="Buscar en los datos..." className="input-field py-2 px-3 text-sm w-64" />
            </div>
          </div>
          <div className="mb-4 flex items-center justify-between">
            <div>
              {filtroBusqueda && (
                <div className="px-4 py-2 bg-blue-50 border-2 border-blue-200 rounded-xl text-sm text-blue-700 font-semibold inline-block">
                  Mostrando {vistaPreviaFiltrada.length} de {vistaPrevia.length} filas
                </div>
              )}
              <div className="mt-2 px-4 py-2 bg-slate-100 border-2 border-slate-200 rounded-xl text-sm text-slate-700 font-semibold inline-block ml-2">
                Página {paginaActual} de {totalPaginas} ({totalFilas} filas totales)
              </div>
            </div>
          </div>
          {Array.isArray(vistaPreviaFiltrada) && vistaPreviaFiltrada.length > 0 ? (
            <>
              <div id="vista-previa-table" className="overflow-auto rounded-xl border-2 border-slate-100 max-h-96">
                <table className="w-full text-sm">
                  <thead className="bg-slate-50 sticky top-0">
                    <tr>
                      {vistaPrevia[0] && Object.keys(vistaPrevia[0]).map((col) => (
                        <th key={col} className="px-4 py-3 text-left font-semibold text-slate-700 whitespace-nowrap">{col}</th>
                      ))}
                    </tr>
                    </thead>
                  <tbody className="divide-y divide-slate-100">
                    {vistaPreviaFiltrada.map((fila, idx) => (
                      <tr key={idx} className="hover:bg-slate-50 transition-colors">
                        {Object.values(fila).map((valor: any, colIdx) => (
                          <td key={colIdx} className={`px-4 py-3 whitespace-nowrap ${String(valor) === '[NULL]' ? 'text-rose-600 font-semibold bg-rose-50' : 'text-slate-600'}`}>
                            {String(valor)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="flex items-center justify-between mt-6 p-4 bg-slate-50 rounded-xl border-2 border-slate-100">
                <div className="flex gap-2">
                  <button onClick={() => cargarPagina(1)} disabled={paginaActual === 1 || cargandoPagina} className="px-3 py-2 bg-primary-600 hover:bg-primary-700 disabled:bg-slate-300 text-white rounded-lg font-semibold transition-colors">Primera</button>
                  <button onClick={() => cargarPagina(paginaActual - 1)} disabled={paginaActual === 1 || cargandoPagina} className="px-3 py-2 bg-primary-600 hover:bg-primary-700 disabled:bg-slate-300 text-white rounded-lg font-semibold transition-colors">Anterior</button>
                </div>
                <div className="flex items-center gap-2">
                  <input type="number" min="1" max={totalPaginas} value={paginaActual} onChange={(e) => { const num = parseInt(e.target.value); if (num >= 1 && num <= totalPaginas) { cargarPagina(num) } }} className="w-20 px-2 py-2 border-2 border-slate-300 rounded-lg text-center font-semibold" />
                  <span className="text-slate-700 font-semibold">de {totalPaginas}</span>
                </div>
                <div className="flex gap-2">
                  <button onClick={() => cargarPagina(paginaActual + 1)} disabled={paginaActual === totalPaginas || cargandoPagina} className="px-3 py-2 bg-primary-600 hover:bg-primary-700 disabled:bg-slate-300 text-white rounded-lg font-semibold transition-colors">Siguiente</button>
                  <button onClick={() => cargarPagina(totalPaginas)} disabled={paginaActual === totalPaginas || cargandoPagina} className="px-3 py-2 bg-primary-600 hover:bg-primary-700 disabled:bg-slate-300 text-white rounded-lg font-semibold transition-colors">Última</button>
                </div>
              </div>
            </>
          ) : (
            <div className="text-center py-8 bg-slate-50 rounded-xl border-2 border-slate-100">
              <svg className="w-12 h-12 text-slate-300 mx-auto mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
              </svg>
              <p className="text-slate-600 font-medium">No hay datos para mostrar</p>
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {mapaCalor.length > 0 && (
            <div className="card p-6">
              <h3 className="text-xl font-bold text-slate-900 mb-4 flex items-center space-x-2">
                <svg className="w-6 h-6 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
                </svg>
                <span>Mapa de Calor - Precios por Zona</span>
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={mapaCalor}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="zona" stroke="#64748b" />
                  <YAxis stroke="#64748b" />
                  <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '2px solid #e2e8f0', borderRadius: '12px' }} />
                  <Legend />
                  <Bar dataKey="precio_promedio" fill="#ef4444" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {datos3D.length > 0 && (
            <div className="card p-6">
              <h3 className="text-xl font-bold text-slate-900 mb-4 flex items-center space-x-2">
                <svg className="w-6 h-6 text-indigo-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
                </svg>
                <span>Dispersión 3D - Precio vs Área vs Habitaciones</span>
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="area_m2" stroke="#64748b" name="Área m²" type="number" />
                  <YAxis dataKey="precio" stroke="#64748b" name="Precio" type="number" />
                  <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '2px solid #e2e8f0', borderRadius: '12px' }} cursor={{ strokeDasharray: '3 3' }} />
                  <Legend />
                  <Scatter data={datos3D} fill="#6366f1" name="Propiedades" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          )}

          {histogramaPrecio.length > 0 && (
            <div className="card p-6">
              <h3 className="text-xl font-bold text-slate-900 mb-4 flex items-center space-x-2">
                <svg className="w-6 h-6 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                <span>Histograma de Distribución de Precios</span>
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={histogramaPrecio}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="rango" stroke="#64748b" />
                  <YAxis stroke="#64748b" />
                  <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '2px solid #e2e8f0', borderRadius: '12px' }} />
                  <Legend />
                  <Bar dataKey="frecuencia" fill="#10b981" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {boxplotZonas.length > 0 && (
            <div className="card p-6">
              <h3 className="text-xl font-bold text-slate-900 mb-4 flex items-center space-x-2">
                <svg className="w-6 h-6 text-orange-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
                </svg>
                <span>Boxplot de Precios por Zona</span>
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={boxplotZonas}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="zona" stroke="#64748b" />
                  <YAxis stroke="#64748b" />
                  <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '2px solid #e2e8f0', borderRadius: '12px' }} />
                  <Legend />
                  <Bar dataKey="min" fill="#cbd5e1" name="Mínimo" radius={[8, 8, 0, 0]} />
                  <Bar dataKey="q1" fill="#94a3b8" name="Q1" radius={[8, 8, 0, 0]} />
                  <Bar dataKey="mediana" fill="#f97316" name="Mediana" radius={[8, 8, 0, 0]} />
                  <Bar dataKey="q3" fill="#64748b" name="Q3" radius={[8, 8, 0, 0]} />
                  <Bar dataKey="max" fill="#475569" name="Máximo" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {serieTemporal.length > 0 && (
            <div className="card p-6 md:col-span-2">
              <h3 className="text-xl font-bold text-slate-900 mb-4 flex items-center space-x-2">
                <svg className="w-6 h-6 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
                </svg>
                <span>Serie Temporal - Evolución de Precios</span>
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={serieTemporal}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="ano" stroke="#64748b" />
                  <YAxis stroke="#64748b" />
                  <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '2px solid #e2e8f0', borderRadius: '12px' }} />
                  <Legend />
                  <Line type="monotone" dataKey="precio_promedio" stroke="#8b5cf6" strokeWidth={3} dot={{ fill: '#8b5cf6', r: 4 }} name="Precio Promedio" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {columnas.length > 0 && (
            <div className="card p-6">
              <h3 className="text-xl font-bold text-slate-900 mb-4 flex items-center space-x-2">
                <svg className="w-6 h-6 text-rose-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
                <span>Valores Nulos por Columna</span>
              </h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={prepararDatosNulos()}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="nombre" stroke="#64748b" />
                  <YAxis stroke="#64748b" />
                  <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '2px solid #e2e8f0', borderRadius: '12px' }} />
                  <Legend />
                  <Bar dataKey="nulos" stackId="a" fill="#ef4444" radius={[8, 8, 0, 0]} />
                  <Bar dataKey="completos" stackId="a" fill="#10b981" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {distribucionClases.length > 0 && (
            <div className="card p-6">
              <h3 className="text-xl font-bold text-slate-900 mb-4 flex items-center space-x-2">
                <svg className="w-6 h-6 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z" />
                </svg>
                <span>Distribución de Datos</span>
              </h3>
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie data={distribucionClases} dataKey="cantidad" nameKey="clase" cx="50%" cy="50%" outerRadius={80} label>
                    {distribucionClases.map((_item, index) => (
                      <Cell key={`cell-${index}`} fill={COLORES[index % COLORES.length]} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '2px solid #e2e8f0', borderRadius: '12px' }} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          )}

          {columnas.filter(c => c.promedio !== undefined).length > 0 && (
            <div className="card p-6 md:col-span-2">
              <h3 className="text-xl font-bold text-slate-900 mb-4 flex items-center space-x-2">
                <svg className="w-6 h-6 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
                </svg>
                <span>Estadísticas Numéricas</span>
              </h3>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={columnas.filter(c => c.promedio !== undefined).slice(0, 6).map(c => ({ nombre: c.nombre.substring(0, 8), promedio: c.promedio, min: c.min, max: c.max }))}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="nombre" stroke="#64748b" />
                  <YAxis stroke="#64748b" />
                  <Tooltip contentStyle={{ backgroundColor: '#ffffff', border: '2px solid #e2e8f0', borderRadius: '12px' }} />
                  <Legend />
                  <Line type="monotone" dataKey="max" stroke="#ef4444" strokeWidth={2} />
                  <Line type="monotone" dataKey="promedio" stroke="#0ea5e9" strokeWidth={3} />
                  <Line type="monotone" dataKey="min" stroke="#10b981" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        <div className="card p-6">
          <div className="flex items-center space-x-3 mb-6">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
              <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <h2 className="text-2xl font-bold text-slate-900">Estadísticas Detalladas</h2>
          </div>
          <div className="space-y-4 max-h-96 overflow-y-auto">
            {columnas.map((col) => (
              <div key={col.nombre} className="border-2 border-slate-100 rounded-xl p-5 hover:border-primary-200 transition-colors">
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <h3 className="font-bold text-slate-900 text-lg">{col.nombre}</h3>
                    <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-slate-100 text-slate-700 mt-1">{col.tipo}</span>
                  </div>
                  <span className="badge badge-info">{col.valores_unicos} únicos</span>
                </div>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div className="flex items-center space-x-2">
                    <svg className="w-4 h-4 text-rose-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                    <span className="text-slate-600">Nulos:</span>
                    <span className={`font-semibold ${col.valores_nulos > 0 ? 'text-rose-600' : 'text-slate-900'}`}>{col.valores_nulos}</span>
                  </div>
                  {col.promedio !== undefined && (
                    <>
                      <div className="flex items-center space-x-2">
                        <svg className="w-4 h-4 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
                        </svg>
                        <span className="text-slate-600">Promedio:</span>
                        <span className="font-semibold text-slate-900">{col.promedio.toFixed(2)}</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <svg className="w-4 h-4 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                        </svg>
                        <span className="text-slate-600">Mínimo:</span>
                        <span className="font-semibold text-slate-900">{col.min}</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <svg className="w-4 h-4 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                        </svg>
                        <span className="text-slate-600">Máximo:</span>
                        <span className="font-semibold text-slate-900">{col.max}</span>
                      </div>
                    </>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}