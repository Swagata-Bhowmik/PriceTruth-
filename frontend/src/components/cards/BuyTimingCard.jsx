import { Clock, Info, ShoppingCart } from 'lucide-react'
import {
  Bar,
  BarChart,
  LabelList,
  ReferenceLine,
  ResponsiveContainer,
  XAxis,
  YAxis,
} from 'recharts'

import Card from '../Card.jsx'
import Badge from '../Badge.jsx'
import UnavailableState from '../UnavailableState.jsx'
import { cn } from '../../lib/cn.js'

/**
 * BuyTimingCard — the Buy Timing Signal feature card (Req 6.1, 6.2, 6.4, 8.3,
 * 10.1). It is a PURE, PROPS-DRIVEN presentational card: it renders the
 * category-level buy-timing module payload returned by
 * `GET /api/v1/buy-timing/{category}` and holds no data-fetching, loading, or
 * error state of its own (those belong to the dashboard's LoadingSkeleton /
 * ErrorBoundary, per the design). The expected `data` shape is:
 *
 *   {
 *     category, available, level: "category", current_month,
 *     recommendation: "buy_now" | "wait" | null,
 *     best_window: {
 *       month, month_name, relative_price_index,
 *       expected_reduction_pct, sale_event
 *     } | null,
 *     disclosure: string,
 *     message: string,
 *   }
 *
 * Design choices:
 *   - The PRIMARY conclusion is shown first (Req 19.4): a text-labelled `Badge`
 *     ("Buy now" / "Wait") whose color reinforces — never replaces — the label
 *     (Req 19.2). A "wait" adds the seasonal window sentence.
 *   - The category-level + snapshot-data `disclosure` is rendered in an
 *     always-visible `role="note"` region (Req 6.4, 10.1) so the honesty note
 *     is present on the collapsed primary view AND on the unavailable view — it
 *     never hides behind the expand control.
 *   - The expandable DETAIL (Req 8.3) is a small seasonal view built purely
 *     from `best_window` (the endpoint returns the single deepest-discount
 *     window, not a full 12-month profile), so it stays correct with just that.
 */

// Fallback honesty note. The backend always sends `disclosure`, but the card
// guarantees the category-level / snapshot limitation is shown even if a caller
// omits it. It deliberately names both "category" and "snapshot".
const DEFAULT_DISCLOSURE =
  'This recommendation is category-level and is based on snapshot data, not a ' +
  'continuous per-product price history.'

const MESSAGE_UNAVAILABLE_FALLBACK =
  'A timing recommendation is unavailable for this category.'

// Per-recommendation presentation. The text `label` is mandatory (Req 19.2):
// status is conveyed by TEXT + color together, never color alone. `tone`
// selects the badge/accent color; `icon` is decorative (Badge marks it
// aria-hidden). `barFill` colors the seasonal bar in the detail view.
const RECOMMENDATION = {
  buy_now: { label: 'Buy now', tone: 'genuine', icon: ShoppingCart, barFill: '#15803d' },
  wait: { label: 'Wait', tone: 'moderate', icon: Clock, barFill: '#b45309' },
}

/** True when the module has no usable recommendation (Req 6.6). */
function isUnavailable(data) {
  return !data || data.available === false || data.recommendation == null
}

/**
 * The seasonal window sentence shown under a "wait" badge, e.g.
 * "Prices usually lowest in November (Big Billion Days), ~18% off." Robust to a
 * missing sale event or reduction — each clause is dropped when absent.
 */
function formatWaitSummary(bestWindow) {
  if (!bestWindow) return null
  const monthName = bestWindow.month_name || 'a seasonal sale window'
  const event = bestWindow.sale_event ? ` (${bestWindow.sale_event})` : ''
  const off = Number.isFinite(bestWindow.expected_reduction_pct)
    ? `, ~${bestWindow.expected_reduction_pct}% off`
    : ''
  return `Prices usually lowest in ${monthName}${event}${off}.`
}

/** Always-visible honesty note (Req 6.4, 10.1). */
function DisclosureNote({ text }) {
  return (
    <div
      role="note"
      className="mt-2 flex items-start gap-1.5 rounded-md border border-gray-200 bg-gray-50 px-2.5 py-1.5 text-xs leading-snug text-gray-600"
    >
      <Info aria-hidden="true" className="mt-0.5 h-3.5 w-3.5 shrink-0 text-gray-400" />
      <span>{text}</span>
    </div>
  )
}

/**
 * Expandable seasonal detail (Req 8.3). Built entirely from `best_window`: the
 * worded window facts (month, expected reduction, sale event) plus a single-bar
 * recharts view of that window's price index against the category average. A
 * descriptive text alternative is exposed via `role="img"` + `aria-label` so
 * the chart is not conveyed by pixels alone (Req 19.5).
 */
function SeasonalityDetail({ category, bestWindow, message, barFill }) {
  const index = Number.isFinite(bestWindow?.relative_price_index)
    ? bestWindow.relative_price_index
    : null
  const indexPct = index != null ? Math.round(index * 100) : null
  const monthName = bestWindow?.month_name || 'the best window'
  const reduction = bestWindow?.expected_reduction_pct
  const saleEvent = bestWindow?.sale_event

  const altText =
    `In the ${category || 'selected'} category, prices around ${monthName}` +
    (indexPct != null
      ? ` sit at about ${indexPct}% of the typical category price`
      : '') +
    (Number.isFinite(reduction) ? ` (roughly ${reduction}% off)` : '') +
    (saleEvent ? `, near ${saleEvent}` : '') +
    '.'

  const chartData = indexPct != null ? [{ name: monthName, index: indexPct }] : []

  return (
    <div className="space-y-3">
      <dl className="grid grid-cols-[auto,1fr] gap-x-4 gap-y-1">
        <dt className="text-gray-500">Best month</dt>
        <dd className="font-medium text-gray-900">{monthName}</dd>
        <dt className="text-gray-500">Expected reduction</dt>
        <dd className="font-medium text-gray-900">
          {Number.isFinite(reduction) ? `~${reduction}% off` : 'Not estimated'}
        </dd>
        <dt className="text-gray-500">Sale event</dt>
        <dd className="font-medium text-gray-900">{saleEvent || 'No major sale event'}</dd>
      </dl>

      {chartData.length ? (
        <figure className="m-0">
          <div role="img" aria-label={altText} className="h-28 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 12, right: 8, bottom: 0, left: -12 }}>
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis
                  domain={[0, 120]}
                  width={44}
                  tick={{ fontSize: 12 }}
                  tickFormatter={(value) => `${value}%`}
                />
                <ReferenceLine
                  y={100}
                  stroke="#9ca3af"
                  strokeDasharray="4 4"
                  label={{
                    value: 'Category avg',
                    position: 'insideTopRight',
                    fontSize: 10,
                    fill: '#6b7280',
                  }}
                />
                <Bar dataKey="index" fill={barFill} radius={[4, 4, 0, 0]} isAnimationActive={false}>
                  <LabelList
                    dataKey="index"
                    position="top"
                    formatter={(value) => `${value}%`}
                    className="fill-gray-600 text-[10px]"
                  />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <figcaption className="mt-1 text-xs text-gray-500">
            Seasonal price index for the {category || 'selected'} category — lower is cheaper;
            100% is the category average.
          </figcaption>
        </figure>
      ) : null}

      {message ? <p className="text-gray-600">{message}</p> : null}
    </div>
  )
}

export default function BuyTimingCard({ data, className }) {
  const module = data ?? {}
  const {
    category,
    recommendation,
    best_window: bestWindow,
    disclosure,
    message,
  } = module

  const unavailable = isUnavailable(data)
  const rec = recommendation ? RECOMMENDATION[recommendation] : undefined
  const disclosureText = disclosure || DEFAULT_DISCLOSURE

  // PRIMARY conclusion, always visible and rendered before any detail (Req 19.4).
  const primary = (
    <div className="space-y-2">
      {unavailable ? (
        <UnavailableState message={message || MESSAGE_UNAVAILABLE_FALLBACK} />
      ) : rec ? (
        <div className="space-y-1.5">
          <Badge tone={rec.tone} icon={rec.icon}>
            {rec.label}
          </Badge>
          {recommendation === 'wait' ? (
            <p className="text-sm text-gray-700">{formatWaitSummary(bestWindow)}</p>
          ) : message ? (
            <p className="text-sm text-gray-600">{message}</p>
          ) : null}
        </div>
      ) : (
        <UnavailableState message={message || MESSAGE_UNAVAILABLE_FALLBACK} />
      )}

      {/* Honesty note — required even on the primary view (Req 6.4, 10.1). */}
      <DisclosureNote text={disclosureText} />
    </div>
  )

  // Expandable seasonal DETAIL (Req 8.3) — only when we actually have a window.
  const detail =
    !unavailable && bestWindow ? (
      <SeasonalityDetail
        category={category}
        bestWindow={bestWindow}
        message={message}
        barFill={rec?.barFill ?? '#15803d'}
      />
    ) : null

  return (
    <Card
      title="Buy timing"
      tone={unavailable ? undefined : rec?.tone}
      primary={primary}
      className={cn(className)}
    >
      {detail}
    </Card>
  )
}
