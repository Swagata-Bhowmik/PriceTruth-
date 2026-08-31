import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts'
import Card from '../Card.jsx'
import Badge from '../Badge.jsx'
import UnavailableState from '../UnavailableState.jsx'

/**
 * ShrinkflationCard — presentational Shrinkflation Timeline card (Task 17.2).
 *
 * This is a PROPS-DRIVEN, presentational component: it performs no data
 * fetching and does not compute the timeline. It consumes the shrinkflation
 * module payload produced by the backend service (see
 * `app/services/shrinkflation_service.py`) exactly as shaped there and renders
 * it inside the shared expandable `Card`.
 *
 * Expected `data` shapes
 * ----------------------
 * Available:
 *   {
 *     status: "ok",
 *     points: [{
 *       observed_at: string (ISO date),
 *       pack_quantity: number,
 *       pack_unit: string,
 *       selling_price: number,
 *       unit_price: number | null,
 *       source_type: "off" | "cited_public_record" | string,
 *       source_citation: string | null,
 *     }, ...],
 *     total_change: {
 *       period_start: string, period_end: string,
 *       pack_quantity_pct: number | null, unit_price_pct: number | null,
 *     } | null,
 *     message: null,
 *   }
 * Unavailable:
 *   { status: "unavailable", points: [], total_change: null, message: string }
 *   OR { available: false, message: string }
 *
 * Behavior
 * --------
 * - Uses the shared `Card`; the PRIMARY conclusion is always rendered first
 *   (Req 19.4): the total pack-quantity and unit-price percentage change over
 *   the recorded period, or "Single recorded pack size" when only one point
 *   exists.
 * - The expandable DETAIL (Req 8.3) holds a recharts dual-axis line chart of
 *   pack quantity and unit price over time, a per-point source attribution
 *   list (Req 4.4), and a text-alternative data table (Req 19.5) so the
 *   timeline is fully available without the chart.
 * - When the module is unavailable (status !== "ok" or available === false),
 *   the card renders the shared `UnavailableState` with the module message
 *   (Req 8.5).
 */

const MONTHS = [
  'Jan',
  'Feb',
  'Mar',
  'Apr',
  'May',
  'Jun',
  'Jul',
  'Aug',
  'Sep',
  'Oct',
  'Nov',
  'Dec',
]

// Plain-language labels for each recorded point's source (Req 4.4).
const SOURCE_LABELS = {
  off: 'Open Food Facts (crowd-sourced)',
  cited_public_record: 'Cited public record',
}

/** Parse the leading `YYYY-MM-DD` of an ISO date string without timezone drift. */
function isoDateParts(iso) {
  if (typeof iso !== 'string') return null
  const match = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso)
  if (!match) return null
  return { year: Number(match[1]), month: Number(match[2]), day: Number(match[3]) }
}

/** Compact "MMM YYYY" label used for the period summary and chart axis. */
function formatMonthYear(iso) {
  const parts = isoDateParts(iso)
  if (!parts) return iso ?? ''
  return `${MONTHS[parts.month - 1]} ${parts.year}`
}

/** Fuller "D MMM YYYY" label used in the text-alternative table rows. */
function formatFullDate(iso) {
  const parts = isoDateParts(iso)
  if (!parts) return iso ?? ''
  return `${parts.day} ${MONTHS[parts.month - 1]} ${parts.year}`
}

/** Signed, rounded percentage for display, e.g. -25 -> "-25%", 33.3 -> "+33%". */
function formatPct(value) {
  if (value == null || Number.isNaN(value)) return 'n/a'
  const rounded = Math.round(value)
  const sign = rounded > 0 ? '+' : ''
  return `${sign}${rounded}%`
}

/** Rupee-formatted selling price, or an em dash when absent. */
function formatPrice(value) {
  if (value == null || Number.isNaN(value)) return '—'
  return `₹${Number(value).toFixed(2)}`
}

/** Rupee-per-unit price (small values), or an em dash when absent. */
function formatUnitPrice(value, unit) {
  if (value == null || Number.isNaN(value)) return '—'
  return `₹${Number(value).toFixed(3)}/${unit || 'unit'}`
}

function sourceLabel(sourceType) {
  return SOURCE_LABELS[sourceType] ?? (sourceType || 'Unknown source')
}

/** Card accent tone: shrinking pack reads as "inflated" (bad for the shopper). */
function toneForChange(totalChange) {
  if (!totalChange || totalChange.pack_quantity_pct == null) return undefined
  return totalChange.pack_quantity_pct < 0 ? 'inflated' : 'genuine'
}

function badgeLabel(totalChange) {
  if (!totalChange || totalChange.pack_quantity_pct == null) return 'Pack-size change'
  if (totalChange.pack_quantity_pct < 0) return 'Shrinkflation detected'
  if (totalChange.pack_quantity_pct > 0) return 'Pack size increased'
  return 'Pack size unchanged'
}

/** The always-visible primary conclusion (Req 19.4). */
function PrimaryConclusion({ points, totalChange, tone }) {
  if (totalChange) {
    return (
      <div className="space-y-1.5">
        <Badge tone={tone ?? 'neutral'}>{badgeLabel(totalChange)}</Badge>
        <p className="text-sm text-gray-800">
          Pack{' '}
          <span className="font-semibold">{formatPct(totalChange.pack_quantity_pct)}</span>{' '}
          · unit price{' '}
          <span className="font-semibold">{formatPct(totalChange.unit_price_pct)}</span>{' '}
          <span className="text-gray-500">
            since {formatMonthYear(totalChange.period_start)}–
            {formatMonthYear(totalChange.period_end)}
          </span>
        </p>
      </div>
    )
  }

  if (points.length === 1) {
    const only = points[0]
    return (
      <div className="space-y-1.5">
        <Badge tone="neutral">Single record</Badge>
        <p className="text-sm text-gray-800">
          Single recorded pack size:{' '}
          <span className="font-semibold">
            {only.pack_quantity} {only.pack_unit}
          </span>{' '}
          at {formatPrice(only.selling_price)}
        </p>
      </div>
    )
  }

  // Defensive: an "ok" status with no points is not expected from the service.
  return <p className="text-sm text-gray-600">No pack-size points recorded.</p>
}

/** The dual-axis recharts timeline plus its non-chart alternatives (detail). */
function TimelineDetail({ points }) {
  const chartData = points.map((point) => ({
    label: formatMonthYear(point.observed_at),
    pack_quantity: point.pack_quantity,
    unit_price: point.unit_price,
  }))

  return (
    <div className="space-y-4">
      {/* Chart with a descriptive text alternative on the container (Req 19.5).
          The full values also appear in the table below. */}
      <div
        role="img"
        aria-label="Line chart of pack quantity and unit price over time. The same values are listed in the data table below."
      >
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={chartData} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="label" tick={{ fontSize: 12 }} />
            <YAxis
              yAxisId="qty"
              orientation="left"
              tick={{ fontSize: 12 }}
              width={44}
            />
            <YAxis
              yAxisId="price"
              orientation="right"
              tick={{ fontSize: 12 }}
              width={44}
            />
            <Tooltip />
            <Legend />
            <Line
              yAxisId="qty"
              type="monotone"
              dataKey="pack_quantity"
              name="Pack quantity"
              stroke="#4f46e5"
              strokeWidth={2}
              dot
              connectNulls
            />
            <Line
              yAxisId="price"
              type="monotone"
              dataKey="unit_price"
              name="Unit price"
              stroke="#b45309"
              strokeWidth={2}
              dot
              connectNulls
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Text alternative for the chart (Req 19.5): the full timeline as data. */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <caption className="mb-1 text-left text-xs font-semibold text-gray-700">
            Pack-size timeline (text alternative)
          </caption>
          <thead>
            <tr className="border-b border-gray-200 text-gray-500">
              <th scope="col" className="py-1 pr-3 font-medium">
                Date
              </th>
              <th scope="col" className="py-1 pr-3 font-medium">
                Pack quantity
              </th>
              <th scope="col" className="py-1 pr-3 font-medium">
                Selling price
              </th>
              <th scope="col" className="py-1 font-medium">
                Unit price
              </th>
            </tr>
          </thead>
          <tbody>
            {points.map((point, index) => (
              <tr
                key={`${point.observed_at}-${index}`}
                className="border-b border-gray-100 text-gray-800"
              >
                <td className="py-1 pr-3">{formatFullDate(point.observed_at)}</td>
                <td className="py-1 pr-3">
                  {point.pack_quantity} {point.pack_unit}
                </td>
                <td className="py-1 pr-3">{formatPrice(point.selling_price)}</td>
                <td className="py-1">
                  {formatUnitPrice(point.unit_price, point.pack_unit)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Per-point source attribution (Req 4.4). */}
      <div>
        <h4 className="mb-1 text-xs font-semibold text-gray-700">Sources</h4>
        <ul className="space-y-1 text-xs text-gray-600">
          {points.map((point, index) => (
            <li key={`src-${point.observed_at}-${index}`}>
              <span className="text-gray-500">{formatMonthYear(point.observed_at)}:</span>{' '}
              {sourceLabel(point.source_type)}
              {point.source_citation ? ` — ${point.source_citation}` : ''}
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}

export default function ShrinkflationCard({ data, title = 'Shrinkflation' }) {
  const isUnavailable =
    !data || data.available === false || (data.status != null && data.status !== 'ok')

  if (isUnavailable) {
    const message = data?.message ?? 'Pack-size history is unavailable for this product.'
    return (
      <Card title={title} primary={<UnavailableState message={message} />} />
    )
  }

  const points = Array.isArray(data.points) ? data.points : []
  const totalChange = data.total_change ?? null
  const tone = toneForChange(totalChange)
  const hasDetail = points.length > 0

  return (
    <Card
      title={title}
      tone={tone}
      primary={<PrimaryConclusion points={points} totalChange={totalChange} tone={tone} />}
    >
      {hasDetail ? <TimelineDetail points={points} /> : null}
    </Card>
  )
}
