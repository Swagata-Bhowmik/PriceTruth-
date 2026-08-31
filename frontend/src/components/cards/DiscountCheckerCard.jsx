import { useMemo } from 'react'
import { ShieldCheck, AlertCircle, AlertTriangle, Info } from 'lucide-react'
import Card from '../Card.jsx'
import Badge from '../Badge.jsx'
import UnavailableState from '../UnavailableState.jsx'
import ShapWaterfall from './ShapWaterfall.jsx'

const CARD_TITLE = 'True Discount Checker'

// Maps the backend `classification` to a human label, a Badge tone, and a
// decorative icon. The label is the text cue that always accompanies color so
// status is never conveyed by color alone (Req 19.2). `neutral`-toned states
// carry no colored accent strip on the Card.
const CLASSIFICATION_META = {
  genuine: { label: 'Genuine', tone: 'genuine', Icon: ShieldCheck },
  moderate: { label: 'Moderate', tone: 'moderate', Icon: AlertCircle },
  likely_inflated: { label: 'Likely inflated', tone: 'inflated', Icon: AlertTriangle },
  verification_limited: { label: 'Verification limited', tone: 'neutral', Icon: Info },
  scoring_unavailable: { label: 'Scoring unavailable', tone: 'neutral', Icon: Info },
}

const FALLBACK_META = { label: 'Unavailable', tone: 'neutral', Icon: Info }
const DEFAULT_UNAVAILABLE_MESSAGE =
  'Discount analysis is unavailable for this product.'

const inr = new Intl.NumberFormat('en-IN', {
  style: 'currency',
  currency: 'INR',
  maximumFractionDigits: 0,
})

function formatPrice(value) {
  return Number.isFinite(value) ? inr.format(value) : null
}

function formatDiscount(pct) {
  return Number.isFinite(pct) ? `${Math.round(pct)}% off` : null
}

// Impact magnitude for the text breakdown, e.g. -18.5 -> "18.5".
function formatImpact(impact) {
  const magnitude = Math.abs(Number(impact) || 0)
  return Number.isInteger(magnitude) ? String(magnitude) : magnitude.toFixed(1)
}

/**
 * True Discount Checker feature card (Req 2.4, 3.2, 8.3, 19.2, 19.4).
 *
 * Purely presentational and PROPS-DRIVEN: the DashboardPage fetches the
 * discount module payload and passes it as `data`. Three payload shapes are
 * handled:
 *   - Scored          — full result with a genuineness score + SHAP explanation.
 *   - Limited          — `genuineness_score: null` with price context + message.
 *   - Contained/absent — `{ available: false, message }` from the dashboard.
 *
 * The primary conclusion (classification + effective discount + prices) is
 * always shown first (Req 19.4); the SHAP waterfall and its text alternative
 * live in the expandable detail region provided by `Card` (Req 8.3). When no
 * score is available the detail shows `UnavailableState` instead of the chart
 * (Req 2.6, 8.5).
 */
export default function DiscountCheckerCard({ data }) {
  const contributions = useMemo(() => {
    const raw = data?.explanation?.contributions
    if (!Array.isArray(raw)) return []
    // Sort by descending |impact| so the strongest drivers lead in both the
    // chart and the text alternative.
    return [...raw].sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact))
  }, [data])

  // Contained / totally unavailable: no price context to show.
  if (!data || data.available === false) {
    return (
      <Card
        title={CARD_TITLE}
        primary={
          <UnavailableState message={data?.message ?? DEFAULT_UNAVAILABLE_MESSAGE} />
        }
      />
    )
  }

  const {
    displayed_price: displayedPrice,
    reference_price: referencePrice,
    effective_discount_pct: effectiveDiscountPct,
    genuineness_score: score,
    classification,
    explanation,
    message,
  } = data

  const hasScore = typeof score === 'number'
  const meta = CLASSIFICATION_META[classification] ?? FALLBACK_META

  const badgeText = hasScore ? `${meta.label} · ${score}% genuine` : meta.label
  const discountText = formatDiscount(effectiveDiscountPct)
  const displayedText = formatPrice(displayedPrice)
  const referenceText = formatPrice(referencePrice)

  const primary = (
    <div className="space-y-2">
      <Badge tone={meta.tone} icon={meta.Icon}>
        {badgeText}
      </Badge>

      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 text-sm">
        {discountText ? (
          <span className="font-semibold text-gray-900">{discountText}</span>
        ) : null}
        {displayedText ? (
          <span className="text-gray-600">
            <span className="sr-only">You pay </span>
            <span className="font-semibold text-gray-900">{displayedText}</span>
            {referenceText ? (
              <>
                {' '}
                <span className="sr-only">reference price </span>
                <span className="text-gray-400 line-through">{referenceText}</span>
              </>
            ) : null}
          </span>
        ) : null}
      </div>
    </div>
  )

  // Detail: SHAP waterfall + an equivalent text breakdown, or the limited /
  // unavailable message when there is no score to explain.
  const detail = hasScore ? (
    <div className="space-y-3">
      <ShapWaterfall
        baseValue={explanation?.base_value}
        finalScore={explanation?.final_score ?? score}
        contributions={contributions}
      />

      {/* Text alternative for the chart (Req 19.5): the same contributions in
          words, so the explanation is fully available without the chart. */}
      <div>
        <p className="text-xs font-medium text-gray-500">Why this verdict</p>
        <ul aria-label="Feature contributions" className="mt-1 space-y-1">
          {contributions.length === 0 ? (
            <li className="text-xs text-gray-500">
              No feature contributions were provided.
            </li>
          ) : (
            contributions.map((c) => {
              const towardGenuine = c.direction === 'toward_genuine'
              return (
                <li key={c.feature} className="text-xs text-gray-700">
                  <span
                    className={
                      towardGenuine
                        ? 'font-medium text-genuine-fg'
                        : 'font-medium text-inflated-fg'
                    }
                  >
                    {c.feature}
                  </span>{' '}
                  pushed toward {towardGenuine ? 'genuine' : 'inflated'} by{' '}
                  {formatImpact(c.impact)}
                </li>
              )
            })
          )}
        </ul>
      </div>
    </div>
  ) : (
    <UnavailableState
      message={
        message ??
        'A genuineness score is unavailable, so only the price context above is shown.'
      }
    />
  )

  return (
    <Card
      title={CARD_TITLE}
      tone={meta.tone === 'neutral' ? undefined : meta.tone}
      primary={primary}
    >
      {detail}
    </Card>
  )
}
