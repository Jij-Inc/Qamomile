import './Information.css'

function Information() {
  const informationItems = [
    {
      title: 'Repository',
      description: 'Qamomile is an open source project. Visit our GitHub repository to contribute.',
      link: 'https://github.com/Jij-Inc/Qamomile',
      buttonText: 'View on GitHub →'
    },
    {
      title: 'About Jij Inc.',
      description: 'Qamomile is developed and maintained by Jij Inc., a leading quantum computing software company.',
      link: 'https://j-ij.com/',
      buttonText: 'Learn more →'
    }
  ]

  return (
    <section className="information">
      <div className="information-container">
        <h2>Information</h2>
        <div className="info-grid">
          {informationItems.map((item, index) => (
            <a
              href={item.link}
              className="info-card"
              key={index}
              target="_blank"
              rel="noopener noreferrer"
            >
              <div className="info-content">
                <div className="info-icon">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                    <circle cx="12" cy="12" r="12" fill="#8A2BE2" />
                  </svg>
                </div>
                <h3>{item.title}</h3>
                <p>{item.description}</p>
                <span className="info-link-text">
                  {item.buttonText}
                </span>
              </div>
            </a>
          ))}
        </div>
      </div>
    </section>
  )
}

export default Information