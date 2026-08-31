import { Info } from 'lucide-react'
import { cn } from '../lib/cn.js'

/**
 * Per-module "no result" state (Req 8.5). Renders a module's unavailable
 * `message` with a neutral, decorative icon in a low-emphasis container so a
 * missing data source reads as informational rather than as an error.
 */
export default function UnavailableState({ message, className }) {
  return (
    <div
      role="status"
      className={cn(
        'flex items-start gap-2 rounded-md bg-gray-50 p-3 text-sm text-gray-600',
        className,
      )}
    >
      <Info aria-hidden="true" className="mt-0.5 h-5 w-5 shrink-0 text-gray-400" />
      <p>{message}</p>
    </div>
  )
}
