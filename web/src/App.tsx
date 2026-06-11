import { useEffect, useState } from 'react'
import { BrowserRouter, NavLink, Navigate, Route, Routes } from 'react-router-dom'
import Projects from './pages/Projects'
import Chat from './pages/Chat'

function Header() {
  const [healthy, setHealthy] = useState<boolean | null>(null)
  useEffect(() => {
    fetch('/api/health').then((r) => setHealthy(r.ok)).catch(() => setHealthy(false))
  }, [])

  const link = ({ isActive }: { isActive: boolean }) =>
    `px-3 py-1.5 rounded-lg text-sm ${isActive ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-400 hover:text-zinc-200'}`

  return (
    <header className="border-b border-zinc-800 px-6 py-3 flex items-center justify-between">
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-sky-500 to-indigo-600 flex items-center justify-center font-bold text-sm">
            M
          </div>
          <h1 className="font-semibold text-[15px]">MLOps Platform</h1>
        </div>
        <nav className="flex gap-1">
          <NavLink to="/projects" className={link}>Projects</NavLink>
          <NavLink to="/chat" className={link}>Chat</NavLink>
        </nav>
      </div>
      <div className="flex items-center gap-2 text-xs text-zinc-400">
        <span className={`w-2 h-2 rounded-full ${
          healthy === null ? 'bg-zinc-600' : healthy ? 'bg-emerald-500' : 'bg-red-500'}`} />
        {healthy === null ? 'connecting…' : healthy ? 'API connected' : 'API offline'}
      </div>
    </header>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-zinc-950 text-zinc-100">
        <Header />
        <Routes>
          <Route path="/" element={<Navigate to="/projects" replace />} />
          <Route path="/projects" element={<Projects />} />
          <Route path="/chat" element={<Chat />} />
        </Routes>
      </div>
    </BrowserRouter>
  )
}
