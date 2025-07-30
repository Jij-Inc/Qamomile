import { Routes, Route } from 'react-router-dom'
import Header from './components/Header/Header'
import Hero from './components/Hero/Hero'
import Concept from './components/Concept/Concept'
import Features from './components/Features/Features'
import Explore from './components/Explore/Explore'
import Tutorials from './components/Tutorials/Tutorials'
import Updates from './components/Updates/Updates'
import Information from './components/Information/Information'
import Footer from './components/Footer/Footer'
import JijLanding from './pages/JijLanding/JijLanding'

function HomePage() {
  return (
    <>
      <Hero />
      <Concept />
      <Features />
      <Explore />
      <Tutorials />
      <Updates />
      <Information />
    </>
  )
}

function App() {
  return (
    <>
      <Header />
      <main>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/jij" element={<JijLanding />} />
        </Routes>
      </main>
      <Footer />
    </>
  )
}

export default App