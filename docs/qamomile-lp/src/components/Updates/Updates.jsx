import './Updates.css'

function Updates() {
  const updates = [
    {
      date: 'Nov 15, 2024',
      content: 'Qamomile is available on the Qiskit ecosystem.'
    },
    {
      date: 'Oct 11, 2024',
      content: 'v0.4.0 has been released with enhanced QAOA parameter handling capabilities, improved documentation including graph partitioning tutorials, and code maintenance updates.'
    },
    {
      date: 'Aug 25, 2024',
      content: 'v0.3.0 has been released with major QRAO enhancements including space-efficient QRAO implementation, new (3,2) and (2,1)-QRAO variants, efficient SU2 ansatz, Hamiltonian arithmetic operations, and improved documentation.'
    },
    {
      date: 'Aug 16, 2024',
      content: 'Rename to Qamomile and release version 0.2.0.'
    }
  ]

  return (
    <section className="updates">
      <div className="updates-container">
        <h2>Updates</h2>
        <div className="update-list">
          {updates.map((update, index) => (
            <div className="update-item" key={index}>
              <div className="update-date">
                <span>{update.date}</span>
              </div>
              <div className="update-content">
                <p>{update.content}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

export default Updates