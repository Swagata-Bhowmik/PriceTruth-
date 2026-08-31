import { ExternalLink, Trophy } from 'lucide-react'
import Card from '../Card.jsx'
import Badge from '../Badge.jsx'
import UnavailableState from '../UnavailableState.jsx'

/**
 * Cross-Platform Aggregator feature card (Req 7.1-7.5, 8.3, 19.4).
 *
 * A purely presentational, props-driven card that renders the cross-platform
 * module shape produced by `aggregate_cross_platform` (and surfaced verbatim by
 * `GET /api/v1/cross-platform/{id}` and the dashboard's `cross_platform` slot):
 *
 *   {
 *     product_id,
 *     available,             // any platform has data (Req 7.6)
 *     comparison_available,  // two or more platforms have data (Req 7.5)
 *     best_deal_platform,
 *     platforms: [
 *       { platform, price, product_url, genuineness_score?, best_deal? },
 *       ...
 *     ],
 *     message,
 *   }
 *
 * It also accepts the dashboard's contained unavailable slot, which carries
 * only `{ available: false, message }`.
 *
 * Presentation rules:
 *   - PRIMARY conclusion first (Req 19.4): the best deal when a comparison is
 *     possible, otherwise the single platform's price plus the no-comparison
 *     message.
 *   - Expandable DETAIL (Req 8.3, via the shared `Card`): every platform entry
 *     sorted cheapest-first, each with a product link (opens in a new tab,
 *     `rel="noopener noreferrer"`, Req 7.3) and its genuineness score ONLY when
 *     the listing has one (Req 7.4). The best-deal row is marked with a text
 *     label, never color alone (Req 19.2).
 *   - When `available === false`, the card body is an `UnavailableState` with
 *     the module's message (Req 7.6, 8.5).
 */

const TITLE = 'Cross-Platform Prices'
const DEFAULT_UNAVAILABLE_MESSAGE =
  'Cross-platform data is unavailable for this product.'

// Indian Rupee formatting for the displayed prices. Kept local to this
// presentational card (there is no shared money formatter yet).
const priceFormatter = new Intl.NumberFormat('en-IN', {
  style: 'currency',
  currency: 'INR',
  maximumFractionDigits: 2,
})

function formatPrice(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—'
  return priceFormatter.format(value)
}

export default function CrossPlatformCard({ data, className }) {
  // Unavailable: either the full module shape with `available: false` or the
  // dashboard's contained slot `{ available: false, message }` (Req 7.6, 8.5).
  if (!data || data.available === false) {
    return (
      <Card
        title={TITLE}
        className={className}
        primary={
          <UnavailableState
            message={data?.message || DEFAULT_UNAVAILABLE_MESSAGE}
          />
        }
      />
    )
  }

  const {
    comparison_available: comparisonAvailable = false,
    best_deal_platform: bestDealPlatform = null,
    platforms = [],
    message = null,
  } = data

  // Defensive: `available` is true but no entries were provided. Treat as
  // unavailable rather than rendering an empty card.
  if (platforms.length === 0) {
    return (
      <Card
        title={TITLE}
        className={className}
        primary={
          <UnavailableState message={message || DEFAULT_UNAVAILABLE_MESSAGE} />
        }
      />
    )
  }

  // Sort cheapest-first in the card itself so the presentation does not depend
  // on the order the data arrives in (Req 7.2 detail ordering).
  const sorted = [...platforms].sort((a, b) => a.price - b.price)

  const isBestDeal = (entry) =>
    entry.best_deal === true ||
    (comparisonAvailable &&
      bestDealPlatform != null &&
      entry.platform === bestDealPlatform)

  // PRIMARY conclusion (always shown before the detail, Req 19.4).
  let primary
  if (comparisonAvailable) {
    // Best deal reads first (Req 7.2). Prefer the flagged/named winner, and
    // fall back to the cheapest entry after the ascending sort.
    const best = sorted.find(isBestDeal) ?? sorted[0]
    primary = (
      <p className="text-sm text-gray-900">
        <span className="font-semibold">Best deal: </span>
        {`${best.platform} at ${formatPrice(best.price)}`}
      </p>
    )
  } else {
    // Single platform: show its price and state that no comparison is
    // available (Req 7.5).
    const only = sorted[0]
    primary = (
      <div className="space-y-1 text-sm">
        <p className="text-gray-900">
          <span className="font-semibold">{`${only.platform}: `}</span>
          {formatPrice(only.price)}
        </p>
        {message ? <p className="text-gray-600">{message}</p> : null}
      </div>
    )
  }

  // Expandable DETAIL: every platform entry, cheapest-first (Req 8.3).
  const detail = (
    <div>
      <p className="mb-2 text-xs font-medium text-gray-500">
        {sorted.length === 1
          ? 'Price on 1 platform'
          : `Prices on ${sorted.length} platforms, cheapest first`}
      </p>
      <ul role="list" className="divide-y divide-gray-100">
        {sorted.map((entry) => {
          const hasScore =
            entry.genuineness_score !== undefined &&
            entry.genuineness_score !== null
          return (
            <li
              key={entry.platform}
              className="flex flex-wrap items-center gap-x-3 gap-y-1 py-2"
            >
              <span className="font-medium text-gray-900">
                {entry.platform}
              </span>
              <span className="text-gray-900">{formatPrice(entry.price)}</span>
              {isBestDeal(entry) ? (
                <Badge tone="genuine" icon={Trophy}>
                  Best deal
                </Badge>
              ) : null}
              {hasScore ? (
                <span className="text-xs text-gray-600">
                  {`Genuineness ${entry.genuineness_score}/100`}
                </span>
              ) : null}
              <a
                href={entry.product_url}
                target="_blank"
                rel="noopener noreferrer"
                className="ml-auto inline-flex items-center gap-1 rounded text-sm font-medium text-focusring underline underline-offset-2 outline-none focus-visible:ring-2 focus-visible:ring-focusring focus-visible:ring-offset-2"
              >
                {`View on ${entry.platform}`}
                <ExternalLink aria-hidden="true" className="h-3.5 w-3.5" />
              </a>
            </li>
          )
        })}
      </ul>
    </div>
  )

  return (
    <Card title={TITLE} className={className} primary={primary}>
      {detail}
    </Card>
  )
}
