import './Explore.css'

function Explore() {
  const exploreItems = [
    {
      title: 'Quick Start Guide',
      description: 'Installation instructions and a simple example to get you started.',
      link: 'https://jij-inc.github.io/Qamomile/en/quickstart.html'
    },
    {
      title: 'API Reference',
      description: 'Complete documentation of Qamomile\'s API.',
      link: 'https://jij-inc.github.io/Qamomile/en/autoapi/index.html'
    },
    {
      title: 'Advanced Topics',
      description: 'Explore advanced features and optimization techniques.',
      link: 'https://jij-inc.github.io/Qamomile/en/tutorial/qaoa/index_qaoa.html'
    }
  ]

  return (
    <section className="explore">
      <div className="explore-container">
        <h2>Explore</h2>
        <p className="explore-description">
          Explore our documentation to
          <br className="sp-only" />
          dive deeper into Qamomile&apos;s capabilities.
        </p>

        <div className="explore-grid">
          {exploreItems.map((item, index) => (
            <div className="explore-item" key={index}>
              <h3>{item.title}</h3>
              <p>{item.description}</p>
              <a href={item.link} className="see-detail">
                See detail
              </a>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

export default Explore
