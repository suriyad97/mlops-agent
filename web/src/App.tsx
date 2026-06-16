import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import Workspace from './pages/Workspace'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Navigate to="/workspace" replace />} />
        <Route path="/workspace" element={<Workspace />} />
        {/* Legacy redirect */}
        <Route path="/projects" element={<Navigate to="/workspace" replace />} />
        <Route path="/chat" element={<Navigate to="/workspace" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
