import { Routes, Route, Link } from 'react-router-dom'
import SearchPage from './pages/SearchPage.jsx'
import DashboardPage from './pages/DashboardPage.jsx'

/**
 * Application shell: a simple header plus the routed page content.
 *
 * Routes:
 *   /                    -> SearchPage    (real implementation: task 18.1)
 *   /product/:productId  -> DashboardPage (real implementation: task 18.2)
 */
export default function App() {
  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      <header className="border-b border-gray-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-4">
          <Link
            to="/"
            className="text-xl font-bold text-indigo-600 hover:text-indigo-700"
          >
            Price Truth
          </Link>
          <span className="text-sm text-gray-500">
            E-Commerce Transparency
          </span>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-6">
        <Routes>
          <Route path="/" element={<SearchPage />} />
          <Route path="/product/:productId" element={<DashboardPage />} />
        </Routes>
      </main>
    </div>
  )
}
