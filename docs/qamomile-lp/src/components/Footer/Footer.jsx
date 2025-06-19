import './Footer.css'

function Footer() {
  const footerLinks = [
    { title: 'Jij Inc.', href: 'https://www.j-ij.com/' },
    { title: 'GitHub', href: 'https://github.com/Jij-Inc/Qamomile?tab=readme-ov-file' },
    { title: 'Discord', href: 'https://discord.gg/Km5dKF9JjG' }
  ]

  return (
    <footer className="footer">
      <div className="footer-container">
        <div className="footer-content">
          <div className="footer-logo">
            <a href="https://jij-inc.github.io/Qamomile/" className="logo">Qamomile</a>
          </div>
          <div className="footer-links">
            {footerLinks.map((link, index) => (
              <a
                key={index}
                href={link.href}
                className="footer-link"
              >
                {link.title}
              </a>
            ))}
          </div>
        </div>
        <div className="footer-bottom">
          <p className="copyright">
            © {new Date().getFullYear()} Jij Inc. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  )
}

export default Footer
