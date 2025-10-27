import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './componentes/Layout'
import Dashboard from './paginas/Dashboard'
import Datasets from './paginas/Datasets'
import Limpieza from './paginas/Limpieza'
import Entrenamiento from './paginas/Entrenamiento'
import Resultados from './paginas/Resultados'
import Prediccion from './paginas/Prediccion'
import AnalisisAvanzado from './paginas/AnalisisAvanzado'
import AnalisisMercado from './paginas/AnalisisMercado'

function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/limpieza/:id" element={<Limpieza />} />
          <Route path="/entrenamiento" element={<Entrenamiento />} />
          <Route path="/resultados" element={<Resultados />} />
          <Route path="/prediccion" element={<Prediccion />} />
          <Route path="/analisis-avanzado" element={<AnalisisAvanzado />} />
          <Route path="/analisis-mercado" element={<AnalisisMercado />} />
          <Route path="*" element={<Navigate to="/" />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  )
}

export default App