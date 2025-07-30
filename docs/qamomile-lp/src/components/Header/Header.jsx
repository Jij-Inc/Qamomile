import { Link } from 'react-router-dom'
import './Header.css'

function Header() {
  return (
    <header className="header">
      <nav className="nav">
        <Link to="/" className="logo">Qamomile</Link>
        <div className="nav-links">
          <a href="https://jij-inc.github.io/Qamomile/">Docs</a>
          <a href="https://jij-inc.github.io/Qamomile/en/autoapi/index.html">API</a>
          <a href="https://jij-inc.github.io/Qamomile/en/quickstart.html">Guide</a>
          <a href="https://discord.gg/Km5dKF9JjG">Community</a>
          <a href="/Qamomile/jij.html">Jij Inc.</a>
        </div>
      </nav>
    </header>
  )
}

export default Header
