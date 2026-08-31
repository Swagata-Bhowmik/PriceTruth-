import { Component } from 'react'
import { AlertTriangle } from 'lucide-react'

/**
 * Contained render-error boundary (Req 8.5, 15.1). Wrapping each dashboard card
 * in one of these means a single card that throws during render shows a local
 * fallback instead of blanking the whole dashboard; sibling cards keep working.
 *
 * Props:
 *   - children : the guarded subtree.
 *   - fallback : optional custom fallback node rendered instead of the default.
 *   - title / message : optional overrides for the default fallback copy.
 */
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError() {
    return { hasError: true }
  }

  componentDidCatch(error, info) {
    // Keep the failure local; log only in dev to aid debugging.
    if (import.meta.env?.DEV) {
      // eslint-disable-next-line no-console
      console.error('ErrorBoundary caught a render error:', error, info)
    }
  }

  render() {
    if (!this.state.hasError) {
      return this.props.children
    }

    if (this.props.fallback !== undefined) {
      return this.props.fallback
    }

    const {
      title = 'Something went wrong',
      message = 'This section could not be displayed. The rest of the page is unaffected.',
    } = this.props

    return (
      <div
        role="alert"
        className="flex items-start gap-2 rounded-md border border-inflated-border bg-inflated-bg p-3 text-sm text-inflated-fg"
      >
        <AlertTriangle aria-hidden="true" className="mt-0.5 h-5 w-5 shrink-0" />
        <div>
          <p className="font-medium">{title}</p>
          <p>{message}</p>
        </div>
      </div>
    )
  }
}
