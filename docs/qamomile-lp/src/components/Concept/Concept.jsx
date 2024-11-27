import './Concept.css'

function Concept() {
  return (
    <section className="concept">
      <div className="concept-background">
        <div className="blur-circle blue"></div>
        <div className="blur-circle purple"></div>
      </div>
      <div className="concept-container">
        <picture className="concept-image">
          <source
            media="(min-width: 768px)"
            srcSet="/image-Qamomile-Concept-pc.png"
          />
          <source
            media="(max-width: 767px)"
            srcSet="/image-Qamomile-Concept-sp.png"
          />
          <img
            src="/image-Qamomile-Concept-pc.png"
            alt="Qamomile Concept: Mathematical Model to Quantum Algorithm"
            className="concept-img"
          />
        </picture>
      </div>
    </section>
  )
}

export default Concept