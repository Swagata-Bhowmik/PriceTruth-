import { useParams } from 'react-router-dom'

/**
 * Placeholder for the composite product dashboard.
 * The full implementation (product header, disclosure banner, five-card
 * feature grid, data-sources panel) lands in task 18.2.
 */
export default function DashboardPage() {
  const { productId } = useParams()

  return (
    <section>
      <h1 className="text-2xl font-semibold">Product dashboard</h1>
      <p className="mt-2 text-gray-600">
        Dashboard for product{' '}
        <span className="font-mono text-gray-900">{productId}</span> is coming
        soon. This is a placeholder page.
      </p>
    </section>
  )
}
