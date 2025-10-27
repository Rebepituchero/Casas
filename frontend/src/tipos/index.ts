export interface Dataset {
  id: string
  nombre: string
  archivo_url: string
  filas: number
  columnas: number
  fecha_subida: string
  usuario_id: string
  es_limpio?: boolean
  dataset_original_id?: string
}

export interface ColumnaStat {
  nombre: string
  tipo: string
  valores_nulos: number
  valores_unicos: number
  promedio?: number
  min?: number
  max?: number
  desviacion?: number
}

export interface ConfiguracionEntrenamiento {
  dataset_id: string
  columnas_entrada: string[]
  columna_objetivo: string
  tipo_modelo: 'regresion' | 'clasificacion' | 'red_neuronal'
  tasa_aprendizaje: number
  epocas: number
  tamano_lote: number
  validacion_split: number
}

export interface Experimento {
  id: string
  nombre: string
  dataset_id: string
  configuracion: ConfiguracionEntrenamiento
  metricas: any
  estado: 'entrenando' | 'completado' | 'error'
  fecha_creacion: string
  metricas_por_epoca?: MetricasEpoca[]
  matriz_confusion?: number[][]
  importancia_features?: FeatureImportance[]
  curva_roc?: CurvaROC
  distribucion_errores?: number[]
  predicciones_vs_reales?: PrediccionReal[]
  tiempo_por_epoca?: number[]
  columnas_entrada?: string[]
  columna_objetivo?: string
}

export interface MetricasEpoca {
  epoca: number
  perdida_entrenamiento: number
  perdida_validacion: number
  precision_entrenamiento?: number
  precision_validacion?: number
  tiempo?: number
}

export interface FeatureImportance {
  feature: string
  importancia: number
}

export interface CurvaROC {
  fpr: number[]
  tpr: number[]
  auc: number
}

export interface PrediccionReal {
  real: number
  prediccion: number
}

export interface CorrelacionMatrix {
  variables: string[]
  matriz: number[][]
}

export interface DistribucionClases {
  clase: string
  cantidad: number
}

export interface EstadisticasLimpieza {
  filas_originales: number
  filas_limpias: number
  filas_eliminadas: number
  porcentaje_datos_eliminados: number
  nulos_por_columna: Record<string, number>
  total_nulos: number
  duplicados_detectados: number
  columnas_eliminadas: string[]
  nulos_eliminados: number
  duplicados_eliminados: number
}

export interface EstadisticasDatos {
  total_filas: number
  total_columnas: number
  total_nulos: number
  total_duplicados: number
  porcentaje_nulos: number
}

export interface ResultadoLimpieza {
  mensaje: string
  filas_resultantes: number
  estadisticas: EstadisticasLimpieza
  dataset_limpio_id: string
  archivo_url: string
}

export interface DatosPrediccionCasa {
  area_total: number
  num_habitaciones: number
  num_banos: number
  num_pisos: number
  antiguedad: number
  area_jardin: number
  garaje: number
  piscina: number
  zona: number
  calidad_construccion: number
  proximidad_transporte: number
  proximidad_escuelas: number
}

export interface PrediccionResultado {
  precio_predicho: number
  tiempo_vida_estimado: number
  tendencia_precio: 'subida' | 'bajada' | 'estable'
  confianza: number
  factores_importantes: Array<{
    nombre: string
    impacto: number
  }>
}