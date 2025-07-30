import fs from 'fs'
import path from 'path'

const fixPaths = () => {
  // Fix index.html -> landing.html
  const indexPath = path.join('dist', 'index.html')
  let indexContent = fs.readFileSync(indexPath, 'utf-8')
  
  // Remove base path and fix asset paths
  indexContent = indexContent.replace(/\/Qamomile\//g, './')
  
  fs.writeFileSync(path.join('dist', 'landing.html'), indexContent)
  
  // Fix jij.html
  const jijPath = path.join('dist', 'jij.html')
  if (fs.existsSync(jijPath)) {
    let jijContent = fs.readFileSync(jijPath, 'utf-8')
    
    // Remove base path and fix asset paths
    jijContent = jijContent.replace(/\/Qamomile\//g, './')
    
    fs.writeFileSync(jijPath, jijContent)
  }
}

fixPaths()