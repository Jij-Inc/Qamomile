import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import JijApp from './JijApp.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <JijApp />
  </StrictMode>,
)