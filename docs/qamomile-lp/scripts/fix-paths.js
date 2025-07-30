import fs from 'fs'
import path from 'path'

const fixPaths = () => {
  // Fix index.html -> landing.html
  const indexPath = path.join('dist', 'index.html')
  let indexContent = fs.readFileSync(indexPath, 'utf-8')
  
  // アセットパスの修正
  indexContent = indexContent.replace(/\/assets\//g, './assets/')
  indexContent = indexContent.replace(/\/image-/g, './image-')
  indexContent = indexContent.replace(/\/Jij_logo\.svg/g, './Jij_logo.svg')
  indexContent = indexContent.replace(/\/JijZept_logo\.png/g, './JijZept_logo.png')
  
  fs.writeFileSync(path.join('dist', 'landing.html'), indexContent)
  
  // Fix jij.html
  const jijPath = path.join('dist', 'jij.html')
  if (fs.existsSync(jijPath)) {
    let jijContent = fs.readFileSync(jijPath, 'utf-8')
    
    // アセットパスの修正
    jijContent = jijContent.replace(/\/assets\//g, './assets/')
    jijContent = jijContent.replace(/\/Jij_logo\.svg/g, './Jij_logo.svg')
    jijContent = jijContent.replace(/\/JijZept_logo\.png/g, './JijZept_logo.png')
    
    fs.writeFileSync(jijPath, jijContent)
  }
}

fixPaths()