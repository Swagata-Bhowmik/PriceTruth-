import { forwardRef } from 'react'
import { cn } from '../lib/cn.js'

/**
 * A button primitive with a guaranteed, visible keyboard focus indicator
 * (Req 19.3). The `:focus-visible` ring is baked in here (in addition to the
 * global `:focus-visible` outline in index.css) so every interactive control
 * built on top of it shows a clear focus state for keyboard users.
 */
const FocusableControl = forwardRef(function FocusableControl(
  { type = 'button', className, children, ...props },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      className={cn(
        'outline-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focusring focus-visible:ring-offset-2',
        className,
      )}
      {...props}
    >
      {children}
    </button>
  )
})

export default FocusableControl
