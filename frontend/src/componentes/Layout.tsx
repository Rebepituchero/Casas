import { ReactNode } from 'react'
import Navegacion from './Navegacion'

interface LayoutProps {
  children: ReactNode
}

export default function Layout({ children }: LayoutProps) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      <Navegacion />
      <main className="container mx-auto px-6 py-8 max-w-7xl animate-fade-in">
        {children}
      </main>
    </div>
  )
}