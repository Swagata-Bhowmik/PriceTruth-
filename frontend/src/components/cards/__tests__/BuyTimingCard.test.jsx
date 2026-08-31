import { describe, it, expect, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'

// The shared Card animates its expandable detail with framer-motion, which
// drives requestAnimationFrame + DOM measurement that jsdom can't run. Replace
// it with synchronous passthrough elements so expansion is deterministic; the
// production Card still uses the real library. (Mirrors Card.test.jsx.)
vi.mock('framer-motion', async () => {
  const React = await import('react')
  const MOTION_ONLY = new Set([
    'initial',
    'animate',
    'exit',
    'transition',
    'variants',
    'whileHover',
    'whileTap',
    'whileFocus',
    'whileInView',
    'layout',
    'layoutId',
    'drag',
  ])
  const filterProps = (props) =>
    Object.fromEntries(Object.entries(props).filter(([k]) => !MOTION_ONLY.has(k)))
  const motion = new Proxy(
    {},
    {
      get: (_target, tag) => {
        if (typeof tag !== 'string') return undefined
        return React.forwardRef((props, ref) =>
          React.createElement(tag, { ref, ...filterProps(props) }),
        )
      },
    },
  )
  return {
    __esModule: true,
    motion,
    AnimatePresence: ({ children }) => React.createElement(React.Fragment, null, children),
  }
})

// recharts renders an SVG inside a size-measuring ResponsiveContainer, which is
// 0x0 in jsdom (and noisy). Swap every used primitive for a passthrough <div>
// so the card's OWN markup (labels, the chart's aria-label text alternative) is
// what these behavioral tests assert on.
vi.mock('recharts', async () => {
  const React = await import('react')
  const passthrough = (name) =>
    React.forwardRef((props, ref) =>
      React.createElement('div', { ref, 'data-recharts': name }, props.children),
    )
  return {
    __esModule: true,
    ResponsiveContainer: passthrough('ResponsiveContainer'),
    BarChart: passthrough('BarChart'),
    Bar: passthrough('Bar'),
    XAxis: passthrough('XAxis'),
    YAxis: passthrough('YAxis'),
    ReferenceLine: passthrough('ReferenceLine'),
    LabelList: passthrough('LabelList'),
    Tooltip: passthrough('Tooltip'),
    Cell: passthrough('Cell'),
  }
})

// Import after the mocks are registered (vi.mock is hoisted, so this is safe).
const { default: BuyTimingCard } = await import('../BuyTimingCard.jsx')

// The real category-level + snapshot-data disclosure the backend attaches to
// every buy-timing result (names both "category" and "snapshot").
const DISCLOSURE =
  'This buy-timing recommendation is category-level - it applies to the product ' +
  'category as a whole, not to an individual product on a single future date - ' +
  'and is derived from point-in-time snapshot data rather than a continuous ' +
  'per-product price history.'

const waitData = {
  category: 'electronics/headphones',
  available: true,
  level: 'category',
  current_month: 8,
  recommendation: 'wait',
  best_window: {
    month: 11,
    month_name: 'November',
    relative_price_index: 0.82,
    expected_reduction_pct: 18,
    sale_event: 'Big Billion Days',
  },
  disclosure: DISCLOSURE,
  message:
    'Prices in the electronics/headphones category have historically been lowest ' +
    'around November (Big Billion Days).',
}

const buyNowData = {
  category: 'grocery/atta',
  available: true,
  level: 'category',
  current_month: 11,
  recommendation: 'buy_now',
  best_window: {
    month: 11,
    month_name: 'November',
    relative_price_index: 0.9,
    expected_reduction_pct: 10,
    sale_event: 'Diwali',
  },
  disclosure: DISCLOSURE,
  message: 'The grocery/atta category is in its historically lowest-price window.',
}

const unavailableData = {
  category: 'unknown-category',
  available: false,
  level: 'category',
  current_month: 5,
  recommendation: null,
  best_window: null,
  disclosure: DISCLOSURE,
  message: 'A timing recommendation is unavailable for this category.',
}

describe('BuyTimingCard (Req 6.1, 6.2, 6.4, 8.3, 10.1)', () => {
  it('shows a "Wait" recommendation with the seasonal window and sale event (Req 6.1, 6.2)', () => {
    render(<BuyTimingCard data={waitData} />)

    // Primary conclusion: a text-labelled "Wait" badge (Req 19.2).
    expect(screen.getByText('Wait')).toBeInTheDocument()

    // The window (month) + the named Indian sale event are shown up front,
    // without needing to expand the card (Req 6.2).
    expect(
      screen.getByText(/lowest in November \(Big Billion Days\), ~18% off/i),
    ).toBeInTheDocument()
  })

  it('shows a "Buy now" recommendation (Req 6.1)', () => {
    render(<BuyTimingCard data={buyNowData} />)

    expect(screen.getByText('Buy now')).toBeInTheDocument()
    // No "wait" is implied.
    expect(screen.queryByText('Wait')).not.toBeInTheDocument()
  })

  it('always shows the category-level + snapshot-data disclosure on the primary view (Req 6.4, 10.1)', () => {
    render(<BuyTimingCard data={waitData} />)

    // The honesty note is present even while the card is collapsed.
    const note = screen.getByRole('note')
    expect(note).toHaveTextContent(/category/i)
    expect(note).toHaveTextContent(/snapshot/i)
  })

  it('still shows the disclosure alongside the unavailable message (Req 6.6, 10.1)', () => {
    render(<BuyTimingCard data={unavailableData} />)

    // Unavailable message is rendered...
    expect(screen.getByText(/unavailable for this category/i)).toBeInTheDocument()
    // ...and the honesty note still holds regardless of availability.
    const note = screen.getByRole('note')
    expect(note).toHaveTextContent(/category/i)
    expect(note).toHaveTextContent(/snapshot/i)
    // No recommendation badge in the unavailable state.
    expect(screen.queryByText('Buy now')).not.toBeInTheDocument()
    expect(screen.queryByText('Wait')).not.toBeInTheDocument()
  })

  it('expands to a seasonal detail view built from best_window (Req 8.3)', () => {
    render(<BuyTimingCard data={waitData} />)

    // Detail is hidden until the card is expanded.
    expect(screen.queryByText(/Best month/i)).not.toBeInTheDocument()

    // Activating the disclosure control expands the detail (Req 8.3).
    fireEvent.click(screen.getByRole('button', { name: /buy timing/i }))

    // The seasonal detail's worded facts appear — these labels are unique to
    // the expanded detail (month, expected reduction, sale event).
    expect(screen.getByText(/Best month/i)).toBeInTheDocument()
    expect(screen.getByText(/Expected reduction/i)).toBeInTheDocument()
    expect(screen.getByText(/Sale event/i)).toBeInTheDocument()
    // The chart carries a descriptive text alternative naming the window and
    // the sale event (Req 19.5), built purely from best_window.
    expect(
      screen.getByRole('img', { name: /around November.*Big Billion Days/i }),
    ).toBeInTheDocument()
  })

  it('renders the fallback disclosure and unavailable message when data is missing', () => {
    render(<BuyTimingCard data={undefined} />)

    const note = screen.getByRole('note')
    expect(note).toHaveTextContent(/category/i)
    expect(note).toHaveTextContent(/snapshot/i)
    expect(screen.getByRole('status')).toBeInTheDocument()
  })
})
