import './Tutorials.css'

function Tutorials() {
  const tutorials = [
    {
      title: 'Quantum Approximate Optimization',
      description: 'Learn how to solve optimization problems.',
      link: 'https://jij-inc.github.io/Qamomile/tutorial/maxcut.html'
    },
    {
      title: 'Quantum Alternating Ansatz',
      description: 'Explore advanced quantum algorithms.',
      link: 'https://jij-inc.github.io/Qamomile/tutorial/alternating_ansatz_graph_coloring.html'
    },
    {
      title: 'Quantum Random Access Optimization',
      description: 'Master quantum optimization techniques.',
      link: 'https://jij-inc.github.io/Qamomile/tutorial/qrao_tutorial.html'
    }
  ]

  return (
    <section className="tutorials">
      <div className="tutorials-background">
        <div className="blur-circle purple bottom"></div>
      </div>
      <div className="tutorials-container">
        <h2>Tutorials</h2>
        <p className="tutorials-description">
          Step-by-step guides and examples to get you started
        </p>

        <div className="tutorials-grid">
          {tutorials.map((tutorial, index) => (
            <div className="tutorial-card" key={index}>
              <div className="tutorial-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="12" fill="#8A2BE2" />
                </svg>
              </div>
              <h3>{tutorial.title}</h3>
              <p>{tutorial.description}</p>
              <a href={tutorial.link} className="tutorial-link">
                Learn more <span className="arrow">→</span>
              </a>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

export default Tutorials