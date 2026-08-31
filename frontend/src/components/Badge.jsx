import { cn } from '../lib/cn.js'

// Literal (not dynamically composed) class strings so Tailwind's JIT keeps
// them. Each tone pairs a light background with a dark same-hue text shade for
// AA contrast (Req 19.1) and always carries a text label (Req 19.2) so status
// is never conveyed by color alone.
const TONE_CLASSES = {
  genuine: 'bg-genuine-bg text-genuine-fg border-genuine-border',
  moderate: 'bg-moderate-bg text-moderate-fg border-moderate-border',
  inflated: 'bg-inflated-bg text-inflated-fg border-inflated-border',
  neutral: 'bg-gray-100 text-gray-800 border-gray-300',
}

/**
 * A small labelled status badge. `children` is the required text label; `tone`
 * selects the color scheme. An optional lucide `icon` is decorative only.
 */
export default function Badge({ tone = 'neutral', icon: Icon, children, className }) {
  const toneClass = TONE_CLASSES[tone] ?? TONE_CLASSES.neutral
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-semibold',
        toneClass,
        className,
      )}
    >
      {Icon ? <Icon aria-hidden="true" className="h-3.5 w-3.5" /> : null}
      <span>{children}</span>
    </span>
  )
}
