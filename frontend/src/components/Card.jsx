import { useId, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { ChevronDown } from 'lucide-react'
import FocusableControl from './FocusableControl.jsx'
import { cn } from '../lib/cn.js'

// Optional colored accent strip per status tone (literal classes for the JIT).
const TONE_ACCENT = {
  genuine: 'bg-genuine',
  moderate: 'bg-moderate',
  inflated: 'bg-inflated',
}

/**
 * Compact, expandable feature card (Req 8.3, 19.4).
 *
 * The `primary` conclusion is always visible and rendered before the detail
 * (Req 19.4). The `children` detail region is revealed when the user hovers the
 * card OR activates the disclosure control (click / Enter / Space) (Req 8.3).
 * The control is a real, tab-focusable <button> exposing `aria-expanded` and
 * `aria-controls`, with a visible focus ring (Req 19.3), so it is fully
 * keyboard accessible. Expansion is animated with framer-motion.
 *
 * Props:
 *   - title    : card heading (also the control's accessible name).
 *   - primary  : the primary conclusion node, always shown.
 *   - children : the detail node, shown when expanded.
 *   - tone     : optional 'genuine' | 'moderate' | 'inflated' accent color.
 */
export default function Card({ title, primary, children, tone, className }) {
  const [pinned, setPinned] = useState(false) // toggled by click / Enter / Space
  const [hovered, setHovered] = useState(false)
  const detailId = useId()

  const hasDetail = children != null && children !== false
  const open = hasDetail && (pinned || hovered)
  const accent = tone ? TONE_ACCENT[tone] : undefined

  return (
    <section
      className={cn(
        'relative overflow-hidden rounded-lg border border-gray-200 bg-white shadow-sm',
        className,
      )}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {accent ? <div aria-hidden="true" className={cn('h-1 w-full', accent)} /> : null}

      <div className="p-4">
        <FocusableControl
          className="flex w-full items-center justify-between gap-2 rounded-md text-left"
          aria-expanded={open}
          aria-controls={hasDetail ? detailId : undefined}
          disabled={!hasDetail}
          onClick={() => setPinned((v) => !v)}
        >
          <h3 className="text-sm font-semibold text-gray-900">{title}</h3>
          {hasDetail ? (
            <ChevronDown
              aria-hidden="true"
              className={cn(
                'h-4 w-4 shrink-0 text-gray-500 transition-transform duration-200',
                open && 'rotate-180',
              )}
            />
          ) : null}
        </FocusableControl>

        {/* Primary conclusion — always visible, before any detail (Req 19.4). */}
        <div className="mt-2">{primary}</div>

        <AnimatePresence initial={false}>
          {open ? (
            <motion.div
              id={detailId}
              key="detail"
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2, ease: 'easeInOut' }}
              className="overflow-hidden"
            >
              <div className="mt-3 border-t border-gray-100 pt-3 text-sm text-gray-700">
                {children}
              </div>
            </motion.div>
          ) : null}
        </AnimatePresence>
      </div>
    </section>
  )
}
