import './Header.css'

function Header() {
  return (
    <header className="header">
      <nav className="nav">
        <a href="/" className="logo">Qamomile</a>
        <div className="nav-links">
          <a href="https://jij-inc.github.io/Qamomile/">Docs</a>
          <a href="https://jij-inc.github.io/Qamomile/en/autoapi/index.html">API</a>
          <a href="https://jij-inc.github.io/Qamomile/en/quickstart.html">Guide</a>
          <a href="https://discord.gg/Km5dKF9JjG">Community</a>
          <a href="https://www.j-ij.com/">Jij Inc.</a>
        </div>
      </nav>
    </header>
  )
}

export default Header