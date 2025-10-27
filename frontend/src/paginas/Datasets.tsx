import { useState, useEffect } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import axios from 'axios'
import { supabase } from '../lib/supabase'
import type { Dataset } from '../tipos'

export default function Datasets() {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [subiendo, setSubiendo] = useState(false)
  const [eliminando, setEliminando] = useState<string | null>(null)
  const [modalEliminar, setModalEliminar] = useState<Dataset | null>(null)
  const navigate = useNavigate()

  useEffect(() => {
    cargarDatasets()
  }, [])

  const cargarDatasets = async () => {
    try {
      const { data } = await axios.get('http://localhost:5000/api/datasets')
      setDatasets(data)
    } catch (error) {
      console.error('Error al cargar datasets:', error)
    }
  }

  const subirArchivo = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const archivo = e.target.files?.[0]
    if (!archivo) return

    setSubiendo(true)
    try {
      const userId = 'demo-user-id'
      const nombreArchivo = `${userId}/${Date.now()}_${archivo.name}`
      const { error: errorSubida } = await supabase.storage
        .from('datasets')
        .upload(nombreArchivo, archivo)

      if (errorSubida) throw errorSubida

      const { data: { publicUrl } } = supabase.storage
        .from('datasets')
        .getPublicUrl(nombreArchivo)

      await axios.post('http://localhost:5000/api/datasets', {
        nombre: archivo.name,
        archivo_url: publicUrl,
        usuario_id: userId,
      })

      cargarDatasets()
    } catch (error) {
      console.error('Error al subir archivo:', error)
    } finally {
      setSubiendo(false)
    }
  }

  const confirmarEliminar = (dataset: Dataset) => {
    setModalEliminar(dataset)
  }

  const eliminarDataset = async () => {
    if (!modalEliminar) return

    setEliminando(modalEliminar.id)
    try {
      await axios.delete(`http://localhost:5000/api/datasets/${modalEliminar.id}`)
      
      const urlParts = new URL(modalEliminar.archivo_url)
      const rutaArchivo = urlParts.pathname.split('/').slice(-2).join('/')
      
      await supabase.storage
        .from('datasets')
        .remove([rutaArchivo])

      cargarDatasets()
      setModalEliminar(null)
    } catch (error) {
      console.error('Error al eliminar dataset:', error)
    } finally {
      setEliminando(null)
    }
  }

  return (
    <div className="space-y-8 animate-slide-up">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-4xl md:text-5xl font-bold text-slate-900 mb-2">Datasets de Casas</h1>
          <p className="text-slate-600 text-lg">Administra tus conjuntos de datos inmobiliarios</p>
        </div>
        <label className="btn-primary cursor-pointer space-x-2">
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
          <span>{subiendo ? 'Subiendo...' : 'Subir Dataset CSV'}</span>
          <input
            type="file"
            accept=".csv"
            onChange={subirArchivo}
            className="hidden"
            disabled={subiendo}
          />
        </label>
      </div>

      {datasets.length === 0 ? (
        <div className="card p-16 text-center">
          <div className="w-32 h-32 mx-auto mb-6 rounded-full bg-gradient-to-br from-slate-100 to-slate-200 flex items-center justify-center">
            <svg className="w-16 h-16 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
            </svg>
          </div>
          <h3 className="text-2xl font-bold text-slate-900 mb-2">No hay datasets de casas</h3>
          <p className="text-slate-600 text-lg mb-6">Sube tu primer archivo CSV con datos inmobiliarios</p>
          <label className="btn-primary inline-flex cursor-pointer space-x-2">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
            <span>Subir Dataset</span>
            <input type="file" accept=".csv" onChange={subirArchivo} className="hidden" disabled={subiendo} />
          </label>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {datasets.map((dataset, index) => (
            <div
              key={dataset.id}
              className="card p-6 transition-all duration-300 group animate-scale-in relative"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  confirmarEliminar(dataset)
                }}
                className="absolute top-4 right-4 w-8 h-8 rounded-lg bg-rose-50 hover:bg-rose-100 text-rose-600 flex items-center justify-center transition-all opacity-0 group-hover:opacity-100 z-10"
                title="Eliminar dataset"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
              </button>

              <div className="cursor-pointer" onClick={() => navigate(`/limpieza/${dataset.id}`)}>
                <div className="flex items-center justify-between mb-6">
                  <div className="relative w-14 h-14">
                    <div className={`absolute inset-0 bg-gradient-to-br ${dataset.es_limpio ? 'from-emerald-600 to-teal-600' : 'from-blue-600 to-cyan-600'} rounded-2xl rotate-3 group-hover:rotate-6 transition-transform`}></div>
                    <div className={`relative w-full h-full bg-gradient-to-br ${dataset.es_limpio ? 'from-emerald-500 to-teal-500' : 'from-blue-500 to-cyan-500'} rounded-2xl flex items-center justify-center shadow-lg`}>
                      <svg className="w-7 h-7 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        {dataset.es_limpio ? (
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        ) : (
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
                        )}
                      </svg>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {dataset.es_limpio && (
                      <span className="badge badge-success text-xs">Limpio</span>
                    )}
                    <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></div>
                  </div>
                </div>

                <h3 className="text-xl font-bold text-slate-900 mb-4 truncate group-hover:text-primary-600 transition-colors">
                  {dataset.nombre}
                </h3>

                <div className="space-y-3 mb-5">
                  <div className="flex items-center justify-between py-2.5 px-4 bg-slate-50 rounded-xl group-hover:bg-blue-50 transition-colors">
                    <div className="flex items-center space-x-2 text-slate-600">
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
                      </svg>
                      <span className="text-sm font-medium">Propiedades</span>
                    </div>
                    <span className="text-sm font-bold text-slate-900">{dataset.filas.toLocaleString()}</span>
                  </div>

                  <div className="flex items-center justify-between py-2.5 px-4 bg-slate-50 rounded-xl group-hover:bg-blue-50 transition-colors">
                    <div className="flex items-center space-x-2 text-slate-600">
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 4v16m6-16v16m-9-9h12" />
                      </svg>
                      <span className="text-sm font-medium">Atributos</span>
                    </div>
                    <span className="text-sm font-bold text-slate-900">{dataset.columnas}</span>
                  </div>

                  <div className="flex items-center justify-between py-2.5 px-4 bg-slate-50 rounded-xl group-hover:bg-blue-50 transition-colors">
                    <div className="flex items-center space-x-2 text-slate-600">
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                      </svg>
                      <span className="text-sm font-medium">Cargado</span>
                    </div>
                    <span className="text-sm font-bold text-slate-900">
                      {new Date(dataset.fecha_subida).toLocaleDateString('es-ES')}
                    </span>
                  </div>
                </div>

                <button className="w-full py-3.5 bg-gradient-to-r from-primary-600 to-purple-600 text-white rounded-xl font-semibold group-hover:shadow-lg transition-all duration-300 flex items-center justify-center space-x-2">
                  <span>Explorar y Limpiar</span>
                  <svg className="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </button>
              </div>

              <Link
                to={`/analisis-avanzado?dataset=${dataset.id}`}
                onClick={(e) => e.stopPropagation()}
                className="w-full py-3.5 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-xl font-semibold hover:shadow-lg transition-all duration-300 flex items-center justify-center space-x-2 mt-3"
              >
                <span>Análisis Avanzado</span>
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </Link>

              <Link
                to={`/analisis-mercado?dataset=${dataset.id}`}
                onClick={(e) => e.stopPropagation()}
                className="w-full py-3.5 bg-gradient-to-r from-emerald-600 to-teal-600 text-white rounded-xl font-semibold hover:shadow-lg transition-all duration-300 flex items-center justify-center space-x-2 mt-3"
              >
                <span>Análisis de Mercado</span>
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </Link>
            </div>
          ))}
        </div>
      )}

      {modalEliminar && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 animate-fade-in">
          <div className="bg-white rounded-2xl p-8 max-w-md w-full mx-4 shadow-2xl animate-scale-in">
            <div className="w-16 h-16 rounded-full bg-rose-100 flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-rose-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </div>
            <h3 className="text-2xl font-bold text-slate-900 text-center mb-2">¿Eliminar Dataset?</h3>
            <p className="text-slate-600 text-center mb-6">
              Se eliminará <span className="font-semibold text-slate-900">{modalEliminar.nombre}</span> y todos los modelos asociados. Esta acción no se puede deshacer.
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
                onClick={eliminarDataset}
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