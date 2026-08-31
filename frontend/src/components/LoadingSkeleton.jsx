import Skeleton from 'react-loading-skeleton'
import 'react-loading-skeleton/dist/skeleton.css'
import { cn } from '../lib/cn.js'

/**
 * Per-module loading state (Req 8.4). Wraps react-loading-skeleton and exposes
 * an accessible live region so assistive tech announces the pending state while
 * a slow module's request is in flight (without blocking sibling modules).
 */
export default function LoadingSkeleton({
  lines = 3,
  className,
  label = 'Loading',
}) {
  return (
    <div
      role="status"
      aria-live="polite"
      aria-busy="true"
      aria-label={label}
      className={cn('space-y-2', className)}
    >
      <Skeleton count={lines} />
      <span className="sr-only">{label}…</span>
    </div>
  )
}
