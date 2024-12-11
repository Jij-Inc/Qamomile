import './Features.css'

function Features() {
  const features = [
    {
      title: 'Versatile Compatibility',
      description: 'Connect to various circuit-level quantum SDKs.'
    },
    {
      title: 'Advanced Algorithm Support',
      description: 'Comprehensive support for quantum algorithms and optimizations.'
    },
    {
      title: 'Flexible Model Conversion',
      description: 'Efficiently convert classical models to quantum circuits.'
    },
    {
      title: 'Intermediate Representation',
      description: 'Clear and organized IR for quantum operations.'
    },
    {
      title: 'Speculative Functionality',
      description: 'Advanced features for quantum experimentation.'
    }
  ]

  return (
    <section className="features">
      <div className="features-container">
        <div className="features-grid">
          {features.map((feature, index) => (
            <div className="feature-card" key={index}>
              <div className="feature-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="12" fill="#8A2BE2" />
                </svg>
              </div>
              <h3>{feature.title}</h3>
              <p>{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

export default Features