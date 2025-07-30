// vite.config.js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

export default defineConfig({
  base: '/Qamomile/',
  plugins: [react()],
  build: {
    assetsInlineLimit: 0, // 画像のインライン化を防ぐ
    // CSS をインライン化
    cssCodeSplit: false,
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        jij: resolve(__dirname, 'jij.html')
      },
      output: {
        // すべてのJSを1ファイルにまとめる
        manualChunks: undefined,
        // ファイル名からハッシュを除去
        entryFileNames: 'assets/[name].js',
        chunkFileNames: 'assets/[name].js',
        assetFileNames: 'assets/[name].[ext]'
      }
    },
    // ソースマップを無効化
    sourcemap: false,
    // minifyを有効化
    minify: true
  }
})