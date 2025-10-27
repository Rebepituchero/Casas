import { useState, useEffect } from 'react'
import axios from 'axios'
import { Link } from 'react-router-dom'

interface EstadisticasDashboard {
  total_datasets: number
  total_experimentos: number
  experimentos_activos: number
  ultimo_entrenamiento?: string
}

export default function Dashboard() {
  const [estadisticas, setEstadisticas] = useState<EstadisticasDashboard | null>(null)
  const [experimentosRecientes, setExperimentosRecientes] = useState([])
  const [eliminando, setEliminando] = useState<string | null>(null)

  useEffect(() => {
    cargarDatos()
  }, [])

  const cargarDatos = async () => {
    try {
      const [respEst, respExp] = await Promise.all([
        axios.get('http://localhost:5000/api/estadisticas'),
        axios.get('http://localhost:5000/api/experimentos/recientes'),
      ])
      setEstadisticas(respEst.data)
      setExperimentosRecientes(respExp.data)
    } catch (error) {
      console.error('Error al cargar datos:', error)
    }
  }

  const eliminarExperimento = async (id: string) => {
    if (!confirm('¿Estás seguro de eliminar este modelo?')) return
    
    setEliminando(id)
    try {
      await axios.delete(`http://localhost:5000/api/experimentos/${id}`)
      cargarDatos()
    } catch (error) {
      console.error('Error al eliminar experimento:', error)
    } finally {
      setEliminando(null)
    }
  }

  const caracteristicas = [
    {
      titulo: 'Gestión de Datos',
      descripcion: 'Carga y administra datasets de propiedades inmobiliarias',
      icon: (
        <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
        </svg>
      ),
      color: 'from-blue-500 to-cyan-500',
      link: '/datasets'
    },
    {
      titulo: 'Limpieza Inteligente',
      descripcion: 'Prepara y optimiza tus datos automáticamente',
      icon: (
        <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      ),
      color: 'from-emerald-500 to-teal-500',
      link: '/datasets'
    },
    {
      titulo: 'Entrenamiento IA',
      descripcion: 'Entrena modelos avanzados de Machine Learning',
      icon: (
        <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
      ),
      color: 'from-purple-500 to-pink-500',
      link: '/entrenamiento'
    },
    {
      titulo: 'Predicciones Precisas',
      descripcion: 'Obtén estimaciones confiables de precios',
      icon: (
        <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
      ),
      color: 'from-amber-500 to-orange-500',
      link: '/prediccion'
    },
    {
      titulo: 'Análisis Avanzado',
      descripcion: 'Visualizaciones y métricas detalladas',
      icon: (
        <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      ),
      color: 'from-rose-500 to-pink-500',
      link: '/analisis-avanzado'
    },
    {
      titulo: 'Análisis de Mercado',
      descripcion: 'Identifica oportunidades de inversión',
      icon: (
        <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      ),
      color: 'from-indigo-500 to-purple-500',
      link: '/analisis-mercado'
    }
  ]

  const tarjetasEstadisticas = [
    {
      titulo: 'Datasets',
      valor: estadisticas?.total_datasets || 0,
      icon: (
        <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
        </svg>
      ),
      gradient: 'from-blue-500 to-cyan-500',
      bgColor: 'bg-blue-50',
      textColor: 'text-blue-600'
    },
    {
      titulo: 'Modelos',
      valor: estadisticas?.total_experimentos || 0,
      icon: (
        <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
      ),
      gradient: 'from-purple-500 to-pink-500',
      bgColor: 'bg-purple-50',
      textColor: 'text-purple-600'
    },
    {
      titulo: 'Activos',
      valor: estadisticas?.experimentos_activos || 0,
      icon: (
        <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
      ),
      gradient: 'from-amber-500 to-orange-500',
      bgColor: 'bg-amber-50',
      textColor: 'text-amber-600'
    }
  ]

  return (
    <div className="space-y-12 animate-slide-up">
      <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-primary-600 via-purple-600 to-pink-600 p-12 shadow-2xl">
        <div className="absolute inset-0 bg-black/10"></div>
        <div className="absolute top-0 right-0 w-1/2 h-full opacity-10">
          <svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
            <path fill="#FFFFFF" d="M44.7,-76.4C58.8,-69.2,71.8,-59.1,79.6,-45.8C87.4,-32.6,90,-16.3,88.5,-0.9C87,14.6,81.4,29.2,73.1,42.8C64.8,56.4,53.8,69,40.4,76.1C27,83.2,11.2,84.8,-4.3,83.5C-19.8,82.2,-35.1,78,-48.6,70.4C-62.1,62.8,-73.8,51.8,-81.2,38.4C-88.6,25,-91.7,9.2,-89.5,-5.6C-87.3,-20.4,-79.8,-34.2,-69.8,-45.8C-59.8,-57.4,-47.3,-66.8,-33.5,-74.3C-19.7,-81.8,-4.6,-87.4,9.4,-85.8C23.4,-84.2,30.6,-83.6,44.7,-76.4Z" transform="translate(100 100)" />
          </svg>
        </div>
        
        <div className="relative z-10">
          <div className="flex items-center space-x-4 mb-6">
            <div className="w-16 h-16 rounded-2xl bg-white/20 backdrop-blur-xl flex items-center justify-center">
              <svg className="w-10 h-10 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
              </svg>
            </div>
            <div>
              <h1 className="text-5xl md:text-6xl font-bold text-white mb-2">
                Bienvenido a Home Predictor
              </h1>
              <p className="text-xl text-white/90">
                Sistema Inteligente de Predicción y Análisis Inmobiliario
              </p>
            </div>
          </div>
          
          <p className="text-white/90 text-lg max-w-3xl mb-8">
            Plataforma avanzada de Machine Learning para predecir precios de propiedades, 
            analizar tendencias del mercado y tomar decisiones de inversión inteligentes.
          </p>

          <div className="flex flex-wrap gap-4">
            <Link to="/datasets" className="px-8 py-4 bg-white text-primary-600 rounded-xl font-bold text-lg hover:bg-white/90 transition-all shadow-xl hover:shadow-2xl hover:scale-105 flex items-center space-x-2">
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              <span>Comenzar Ahora</span>
            </Link>
            <Link to="/prediccion" className="px-8 py-4 bg-white/10 backdrop-blur-xl text-white border-2 border-white/30 rounded-xl font-bold text-lg hover:bg-white/20 transition-all flex items-center space-x-2">
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              <span>Hacer Predicción</span>
            </Link>
          </div>
        </div>
      </div>

      <div>
        <div className="text-center mb-8">
          <h2 className="text-3xl md:text-4xl font-bold text-slate-900 mb-3">
            Estadísticas del Sistema
          </h2>
          <p className="text-slate-600 text-lg">
            Resumen de tu actividad y recursos disponibles
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {tarjetasEstadisticas.map((tarjeta, index) => (
            <div
              key={index}
              className="stat-card group animate-scale-in"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <div className="flex items-start justify-between mb-4">
                <div className={`w-16 h-16 rounded-2xl ${tarjeta.bgColor} flex items-center justify-center ${tarjeta.textColor} group-hover:scale-110 transition-transform duration-300`}>
                  {tarjeta.icon}
                </div>
              </div>
              <h3 className="text-slate-600 text-sm font-semibold mb-2 uppercase tracking-wide">{tarjeta.titulo}</h3>
              <p className="text-5xl font-bold text-slate-900 mb-4">{tarjeta.valor}</p>
              <div className={`h-2 rounded-full bg-gradient-to-r ${tarjeta.gradient}`}></div>
            </div>
          ))}
        </div>
      </div>

      <div>
        <div className="text-center mb-8">
          <h2 className="text-3xl md:text-4xl font-bold text-slate-900 mb-3">
            Funcionalidades Principales
          </h2>
          <p className="text-slate-600 text-lg">
            Descubre todo lo que puedes hacer con Home Predictor
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {caracteristicas.map((caracteristica, index) => (
            <Link
              key={index}
              to={caracteristica.link}
              className="card p-8 hover:shadow-2xl transition-all duration-300 group cursor-pointer animate-scale-in"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <div className={`w-16 h-16 rounded-2xl bg-gradient-to-br ${caracteristica.color} flex items-center justify-center text-white mb-6 group-hover:scale-110 transition-transform duration-300 shadow-lg`}>
                {caracteristica.icon}
              </div>
              <h3 className="text-2xl font-bold text-slate-900 mb-3 group-hover:text-primary-600 transition-colors">
                {caracteristica.titulo}
              </h3>
              <p className="text-slate-600 leading-relaxed mb-4">
                {caracteristica.descripcion}
              </p>
              <div className="flex items-center text-primary-600 font-semibold group-hover:translate-x-2 transition-transform">
                <span>Explorar</span>
                <svg className="w-5 h-5 ml-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </div>
            </Link>
          ))}
        </div>
      </div>

      {experimentosRecientes.length > 0 && (
        <div className="card p-8">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-purple-500 flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h2 className="text-3xl font-bold text-slate-900">Modelos Recientes</h2>
            </div>
            <Link to="/resultados" className="text-primary-600 hover:text-primary-700 font-semibold flex items-center space-x-1 group">
              <span>Ver todos</span>
              <svg className="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </Link>
          </div>
          <div className="space-y-3">
            {experimentosRecientes.map((exp: any, index) => (
              <div
                key={exp.id}
                className="group flex items-center justify-between p-5 bg-slate-50 rounded-xl hover:bg-gradient-to-r hover:from-blue-50 hover:to-indigo-50 transition-all duration-300 border-2 border-transparent hover:border-primary-200 animate-slide-up relative"
                style={{ animationDelay: `${index * 0.05}s` }}
              >
                <Link to={`/resultados?experimento=${exp.id}`} className="flex items-center space-x-4 flex-1">
                  <div className="relative w-12 h-12">
                    <div className="absolute inset-0 bg-gradient-to-br from-primary-600 to-purple-600 rounded-xl opacity-75"></div>
                    <div className="relative w-full h-full bg-gradient-to-br from-primary-500 to-purple-500 rounded-xl flex items-center justify-center text-white font-bold shadow-lg">
                      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
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
                </Link>
                <div className="flex items-center gap-3">
                  <span className={`badge ${
                    exp.estado === 'completado' ? 'badge-success' :
                    exp.estado === 'entrenando' ? 'badge-warning' : 'badge-error'
                  }`}>
                    {exp.estado}
                  </span>
                  <button
                    onClick={(e) => {
                      e.preventDefault()
                      eliminarExperimento(exp.id)
                    }}
                    disabled={eliminando === exp.id}
                    className="w-8 h-8 rounded-lg bg-rose-50 hover:bg-rose-100 text-rose-600 flex items-center justify-center transition-all opacity-0 group-hover:opacity-100"
                    title="Eliminar modelo"
                  >
                    {eliminando === exp.id ? (
                      <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                    ) : (
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                    )}
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card p-8 bg-gradient-to-br from-blue-50 to-cyan-50 border-2 border-blue-100">
          <div className="w-14 h-14 rounded-xl bg-blue-500 flex items-center justify-center text-white mb-4">
            <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <h3 className="text-xl font-bold text-slate-900 mb-2">Precisión Avanzada</h3>
          <p className="text-slate-600 mb-4">Modelos de Machine Learning entrenados con datos reales para predicciones precisas</p>
          <div className="flex items-center text-blue-600 font-semibold">
            <span>95%+ precisión</span>
          </div>
        </div>

        <div className="card p-8 bg-gradient-to-br from-purple-50 to-pink-50 border-2 border-purple-100">
          <div className="w-14 h-14 rounded-xl bg-purple-500 flex items-center justify-center text-white mb-4">
            <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <h3 className="text-xl font-bold text-slate-900 mb-2">Análisis Profundo</h3>
          <p className="text-slate-600 mb-4">Visualizaciones interactivas y métricas detalladas de mercado inmobiliario</p>
          <div className="flex items-center text-purple-600 font-semibold">
            <span>20+ métricas</span>
          </div>
        </div>

        <div className="card p-8 bg-gradient-to-br from-emerald-50 to-teal-50 border-2 border-emerald-100">
          <div className="w-14 h-14 rounded-xl bg-emerald-500 flex items-center justify-center text-white mb-4">
            <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <h3 className="text-xl font-bold text-slate-900 mb-2">Decisiones Inteligentes</h3>
          <p className="text-slate-600 mb-4">Identifica las mejores oportunidades de inversión inmobiliaria</p>
          <div className="flex items-center text-emerald-600 font-semibold">
            <span>ROI optimizado</span>
          </div>
        </div>
      </div>
    </div>
  )
}