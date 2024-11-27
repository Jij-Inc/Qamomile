import './Hero.css'

function Hero() {
  return (
    <section className="hero">
      <div className="hero-background">
        <div className="blur-circle blue"></div>
        <div className="blur-circle purple"></div>
        <div className="hero-watermark">Welcome to Qamomile</div>
      </div>
      <div className="hero-container">
        <div className="hero-content">
          <h1>Qamomile</h1>
          <p className="hero-subtitle">
            Qamomile is a powerful SDK for quantum optimization algorithms.
          </p>
          <a href="https://github.com/Jij-Inc/Qamomile" className="cta-button">
            Go to Github
          </a>
        </div>
      </div>
      <div className="hero-description">
        <p>
          Qamomile is specializing in the conversion of mathematical models<br />
          into quantum circuits. It serves as a bridge between classical<br />
          optimization problems and quantum computing solutions.
        </p>
      </div>
    </section>
  )
}

export default Hero