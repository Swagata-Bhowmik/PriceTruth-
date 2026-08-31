import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import Card from '../Card.jsx'
import Badge from '../Badge.jsx'
import UnavailableState from '../UnavailableState.jsx'
import { cn } from '../../lib/cn.js'

// Bar fills: the best-value bar uses the "genuine" green accent so the winning
// variant stands out; the rest are neutral gray. Color is only a secondary cue
// here — the best value is also stated in words in the primary conclusion, the
// table, and the chart text alternative (Req 19.5).
const BEST_FILL = '#15803d' // green-700 (genuine.DEFAULT)
const BASE_FILL = '#9ca3af' // gray-400

// Friendly wording for the backend's exclusion reason codes (Req 5.5).
const EXCLUSION_REASONS = {
  non_positive_quantity: 'pack quantity was zero or negative',
  missing_quantity: 'pack quantity was missing',
  invalid_quantity: 'pack quantity was invalid',
  invalid_unit: 'unit of measure was not recognized',
}

// Prices are shown in Indian Rupees — the platform targets Indian e-commerce
// (Indian sale calendar, supported platforms, etc.). The module shape carries
// bare numbers, so formatting lives here in the presentational layer.
function formatPrice(value) {
  const n = Number(value)
  if (!Number.isFinite(n)) return '—'
  return `₹${n.toFixed(2)}`
}

// Unit prices can be very small (e.g. 0.199/g), so show up to 3 decimals and
// trim trailing zeros for readability (0.450 -> ₹0.45, 2.000 -> ₹2).
function formatUnitPrice(value) {
  const n = Number(value)
  if (!Number.isFinite(n)) return '—'
  let s = n.toFixed(3)
  if (s.includes('.')) s = s.replace(/\.?0+$/, '')
  return `₹${s}`
}

function describeReason(reason) {
  if (!reason) return 'excluded from the comparison'
  return EXCLUSION_REASONS[reason] ?? String(reason).replace(/_/g, ' ')
}

/**
 * Unit Price Comparator feature card (Req 5.2, 5.3, 8.3, 19.4, 19.5).
 *
 * A props-driven, presentational card that consumes the unit-price module
 * shape directly, so the dashboard can render it as `<UnitPriceCard {...module} />`:
 *
 *   Available:   { standard_unit: 'g' | 'ml',
 *                  comparison: [{ label, price, quantity_std, unit_price, best_value? }],
 *                  excluded:   [{ label, reason }] }
 *   Unavailable: { available: false, message }
 *
 * The PRIMARY conclusion — the best-value variant — is always shown first, via
 * the shared `Card`, before any detail (Req 19.4). Expanding the card (hover or
 * keyboard, handled by `Card`) reveals the DETAIL (Req 8.3): a Recharts bar
 * chart of unit price per variant with the best-value bar highlighted, a table
 * of each variant's price / pack quantity / unit price (Req 5.3), and any
 * excluded variants with their reason (Req 5.5). The chart is decorative
 * (`aria-hidden`); an equivalent worded summary provides its text alternative
 * so the data is available without the chart (Req 19.5).
 */
export default function UnitPriceCard({
  available = true,
  message,
  standard_unit: standardUnit,
  comparison,
  excluded,
  title = 'Unit price',
  className,
}) {
  const entries = Array.isArray(comparison) ? comparison.filter(Boolean) : []
  const excludedList = Array.isArray(excluded) ? excluded.filter(Boolean) : []

  // No result for this module: render the shared unavailable state in place of
  // the comparison (Req 8.5), still wrapped in a Card for a consistent grid.
  if (available === false || entries.length === 0) {
    const fallback = 'Unit-price comparison is unavailable for this product.'
    return <Card title={title} className={className} primary={<UnavailableState message={message || fallback} />} />
  }

  const unit = standardUnit ?? ''
  const perUnit = unit ? `/${unit}` : ''

  // Best value = the variant the backend flagged, falling back to the minimum
  // unit price if none is flagged (Req 5.2).
  const best =
    entries.find((v) => v.best_value) ??
    entries.reduce((min, v) => (Number(v.unit_price) < Number(min.unit_price) ? v : min), entries[0])

  const primaryText = `Best value: ${best.label} at ${formatUnitPrice(best.unit_price)}${perUnit}`

  // Worded text alternative for the chart — same per-variant data (Req 19.5).
  const chartSummary =
    `Unit price per ${unit || 'unit'} by variant: ` +
    entries
      .map(
        (v) =>
          `${v.label} at ${formatUnitPrice(v.unit_price)}${perUnit}${v.best_value ? ' (best value)' : ''}`,
      )
      .join('; ') +
    '.'

  const detail = (
    <div className="space-y-4">
      {/* Chart (decorative) + its text alternative (Req 19.5). */}
      <figure className="m-0">
        <figcaption className="mb-1 text-xs font-medium text-gray-500">
          Unit price by variant (₹ per {unit || 'unit'})
        </figcaption>
        <div aria-hidden="true" className="h-48 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={entries} margin={{ top: 8, right: 8, bottom: 4, left: 4 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="label" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} width={44} />
              <Tooltip
                formatter={(value) => [`${formatUnitPrice(value)}${perUnit}`, 'Unit price']}
              />
              <Bar dataKey="unit_price" radius={[4, 4, 0, 0]}>
                {entries.map((v) => (
                  <Cell key={v.label} fill={v.best_value ? BEST_FILL : BASE_FILL} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p data-testid="unit-price-text-alt" className="mt-1 text-xs text-gray-600">
          {chartSummary}
        </p>
      </figure>

      {/* Per-variant comparison: price, pack quantity, unit price (Req 5.3). */}
      <table data-testid="unit-price-table" className="w-full border-collapse text-left text-sm">
        <thead>
          <tr className="border-b border-gray-200 text-xs uppercase tracking-wide text-gray-500">
            <th scope="col" className="py-1 pr-2 font-medium">
              Variant
            </th>
            <th scope="col" className="py-1 pr-2 font-medium">
              Price
            </th>
            <th scope="col" className="py-1 pr-2 font-medium">
              Pack qty
            </th>
            <th scope="col" className="py-1 pr-2 font-medium">
              Unit price
            </th>
          </tr>
        </thead>
        <tbody>
          {entries.map((v) => (
            <tr
              key={v.label}
              className={cn('border-b border-gray-100', v.best_value && 'bg-genuine-bg')}
            >
              <th scope="row" className="py-1 pr-2 font-medium text-gray-900">
                <span className="inline-flex items-center gap-1.5">
                  {v.label}
                  {v.best_value ? <Badge tone="genuine">Best value</Badge> : null}
                </span>
              </th>
              <td className="py-1 pr-2 tabular-nums text-gray-700">{formatPrice(v.price)}</td>
              <td className="py-1 pr-2 tabular-nums text-gray-700">
                {v.quantity_std}
                {unit ? ` ${unit}` : ''}
              </td>
              <td className="py-1 pr-2 tabular-nums text-gray-900">
                {formatUnitPrice(v.unit_price)}
                {perUnit}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* Variants excluded from the comparison, with the reason (Req 5.5). */}
      {excludedList.length > 0 ? (
        <div data-testid="unit-price-excluded" className="text-xs text-gray-600">
          <p className="font-medium text-gray-700">Excluded from comparison</p>
          <ul className="mt-1 list-disc space-y-0.5 pl-5">
            {excludedList.map((e) => (
              <li key={e.label}>
                <span className="font-medium text-gray-800">{e.label}</span>
                {' — '}
                {describeReason(e.reason)}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  )

  return (
    <Card
      title={title}
      tone="genuine"
      className={className}
      primary={<p className="text-sm font-semibold text-gray-900">{primaryText}</p>}
    >
      {detail}
    </Card>
  )
}
