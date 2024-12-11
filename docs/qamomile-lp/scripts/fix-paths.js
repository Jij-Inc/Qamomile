import fs from 'fs'
import path from 'path'

const fixPaths = () => {
  const htmlPath = path.join('dist', 'index.html')
  let content = fs.readFileSync(htmlPath, 'utf-8')

  // アセットパスの修正
  content = content.replace(/\/assets\//g, './assets/')
  content = content.replace(/\/image-/g, './image-')

  fs.writeFileSync(path.join('dist', 'landing.html'), content)
}

fixPaths()