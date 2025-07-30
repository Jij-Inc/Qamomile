import { motion } from 'framer-motion'
import { useEffect, useState } from 'react'
import JijLogo from '../../assets/Jij_logo.svg'
import JijZeptLogo from '../../assets/JijZept_logo.png'
import './JijLanding.css'

function JijLanding() {
  const [particles, setParticles] = useState([])

  useEffect(() => {
    const generateParticles = () => {
      const newParticles = []
      for (let i = 0; i < 80; i++) {
        newParticles.push({
          id: i,
          x: Math.random() * 100,
          y: Math.random() * 100,
          size: Math.random() * 8 + 3,
          duration: Math.random() * 15 + 5,
          delay: Math.random() * 5,
          moveX: (Math.random() - 0.5) * 150,
          moveY: (Math.random() - 0.5) * 150
        })
      }
      setParticles(newParticles)
    }
    
    generateParticles()
  }, [])

  return (
    <div className="jij-landing">

      <a href="/Qamomile/landing.html" className="back-button">← Back to Qamomile</a>

      <section className="jij-hero">
        <div className="quantum-background">
          {particles.map(particle => (
            <motion.div
              key={particle.id}
              className="quantum-particle"
              style={{
                left: `${particle.x}%`,
                top: `${particle.y}%`,
                width: `${particle.size}px`,
                height: `${particle.size}px`,
              }}
              animate={{
                x: [0, particle.moveX, 0],
                y: [0, particle.moveY, 0],
                opacity: [0.2, 0.6, 0.2],
              }}
              transition={{
                duration: particle.duration,
                repeat: Infinity,
                delay: particle.delay,
                ease: "easeInOut"
              }}
            />
          ))}
          
          <motion.div 
            className="single-orbit-container orbit-right"
            animate={{
              rotate: 360
            }}
            transition={{
              duration: 20,
              repeat: Infinity,
              ease: "linear"
            }}
          >
            <div className="single-orbit">
              <div className="orbit-electron" />
            </div>
          </motion.div>
          
          <motion.div 
            className="single-orbit-container orbit-left"
            animate={{
              rotate: -360
            }}
            transition={{
              duration: 25,
              repeat: Infinity,
              ease: "linear"
            }}
          >
            <div className="single-orbit">
              <div className="orbit-electron" />
            </div>
          </motion.div>
          
          <div className="quantum-grid" />
        </div>

        <motion.div 
          className="jij-hero-content"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <motion.img 
            src={JijLogo}
            alt="Jij"
            className="jij-logo-image"
            initial={{ scale: 0.5, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          />
          <motion.p 
            className="jij-tagline"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.4 }}
          >
            Bridging Mathematical Optimization and Quantum Computing
          </motion.p>
          <motion.p 
            className="jij-mission"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.6 }}
          >
            Make Society Computable, Contributing to the Advancement of Humanity.
          </motion.p>
        </motion.div>
      </section>

      <section className="jij-about">
        <div className="container">
          <motion.div 
            className="about-content"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <h2>Innovative Software Development</h2>
            <p>
              Jij develops advanced software and algorithms that bridge
              mathematical optimization and quantum computing.
            </p>
          </motion.div>
        </div>
      </section>

      <section className="jij-platform">
        <div className="container">
          <motion.div 
            className="jijzept-header"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <img src={JijZeptLogo} alt="JijZept" className="jijzept-logo-img" />
          </motion.div>
          <motion.p 
            className="platform-description"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.1 }}
          >
            Comprehensive Quantum and Optimization Software Stack
          </motion.p>
          
          <motion.div 
            className="platform-stack"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <div className="stack-layer application-layer">
              <span className="layer-label">Application</span>
            </div>
            
            <div className="stack-components">
              <motion.div 
                className="stack-item sdk"
                whileHover={{ scale: 1.02 }}
              >
                <div className="item-header">
                  <h3>JijZept SDK</h3>
                  <span className="badge oss">OSS / Free</span>
                </div>
                <p>Quantum and Optimization SDK</p>
              </motion.div>
              
              <motion.div 
                className="stack-item tools"
                whileHover={{ scale: 1.02 }}
              >
                <div className="item-header">
                  <h3>JijZept Tools</h3>
                  <span className="badge commercial">COMMERCIAL</span>
                </div>
                <p>Algorithm Development Tools for Quantum and Classical Optimization</p>
              </motion.div>
              
              <div className="stack-row">
                <motion.div 
                  className="stack-item solver"
                  whileHover={{ scale: 1.02 }}
                >
                  <div className="item-header">
                    <h3>JijZept Solver</h3>
                    <span className="badge commercial">COMMERCIAL</span>
                  </div>
                  <p>Useful Optimization Solver</p>
                </motion.div>
                
                <motion.div 
                  className="stack-item qamomile"
                  whileHover={{ scale: 1.02 }}
                >
                  <div className="item-header">
                    <h3>Qamomile</h3>
                    <span className="badge oss">OSS</span>
                  </div>
                  <p>Classical → Quantum Encoder</p>
                </motion.div>
              </div>
            </div>
            
            <div className="hardware-layers">
              <div className="hardware-layer classical">
                <span>Classical Hardware</span>
              </div>
              <div className="hardware-layer quantum">
                <span>Quantum Hardware</span>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      <section className="jij-features">
        <div className="container">
          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            Our Technology
          </motion.h2>
          
          <div className="features-grid">
            <motion.div 
              className="feature-card"
              initial={{ opacity: 0, x: -30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.1 }}
              whileHover={{ scale: 1.05 }}
            >
              <div className="feature-icon quantum-icon">
                <div className="quantum-orbit" />
                <div className="quantum-nucleus" />
              </div>
              <h3>Quantum Computing</h3>
              <p>Revolutionary software development for quantum computers</p>
            </motion.div>

            <motion.div 
              className="feature-card"
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.2 }}
              whileHover={{ scale: 1.05 }}
            >
              <div className="feature-icon optimization-icon">
                <div className="optimization-graph" />
              </div>
              <h3>Mathematical Optimization</h3>
              <p>Advanced algorithms solving complex optimization problems</p>
            </motion.div>

            <motion.div 
              className="feature-card"
              initial={{ opacity: 0, x: 30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.3 }}
              whileHover={{ scale: 1.05 }}
            >
              <div className="feature-icon innovation-icon">
                <div className="innovation-spark" />
              </div>
              <h3>Advanced Innovation</h3>
              <p>Forward-looking advanced technology development</p>
            </motion.div>
          </div>
        </div>
      </section>

      <section className="jij-cta">
        <motion.div 
          className="cta-content"
          initial={{ opacity: 0, scale: 0.9 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <h2>Explore the Future of Computing</h2>
          <p>Join us in revolutionizing optimization with quantum technology</p>
          <div className="cta-buttons">
            <motion.a 
              href="https://www.j-ij.com/" 
              className="cta-button primary"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Visit Jij Website
            </motion.a>
            <motion.a 

              href="/Qamomile/landing.html" 

              className="cta-button secondary"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Explore Qamomile
            </motion.a>
          </div>
        </motion.div>
      </section>
    </div>
  )
}

export default JijLanding