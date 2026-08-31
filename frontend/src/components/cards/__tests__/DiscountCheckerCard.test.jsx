import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

// react-plotly.js is loaded on demand (dynamic import) inside ShapWaterfall and
// pulls in the full Plotly bundle, which jsdom cannot render. Replace it with a
// tiny stub that records the number of traces it received so the chart path is
// exercised without touching real Plotly.
vi.mock('react-plotly.js', () => ({
  __esModule: true,
  default: (props) => (
    <div data-testid="plotly-mock" data-trace-count={props?.data?.length ?? 0} />
  ),
}))

// framer-motion drives real animations (requestAnimationFrame + DOM
// measurement) that jsdom can't run; the Card uses it for expand/collapse.
// Swap in synchronous passthrough elements so the detail region renders
// deterministically when expanded.
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
    AnimatePresence: ({ children }) =>
      React.createElement(React.Fragment, null, children),
  }
})

// Import after the mocks are registered (vi.mock is hoisted).
const { default: DiscountCheckerCard } = await import('../DiscountCheckerCard.jsx')

const scoredData = {
  displayed_price: 1499.0,
  reference_price: 4999.0,
  effective_discount_pct: 70.0,
  genuineness_score: 42,
  classification: 'likely_inflated',
  explanation: {
    base_value: 55.0,
    final_score: 42,
    contributions: [
      {
        feature: "How inflated the 'original' price looks vs. the category",
        impact: -18.5,
        direction: 'toward_inflated',
      },
      {
        feature: 'Size of the claimed discount vs. the category norm',
        impact: -9.2,
        direction: 'toward_inflated',
      },
      { feature: 'Review volume', impact: 4.1, direction: 'toward_genuine' },
    ],
  },
}

describe('DiscountCheckerCard (Req 2.4, 3.2, 8.3, 19.2, 19.4)', () => {
  it('shows the classification as text (not color alone) plus the effective discount', () => {
    render(<DiscountCheckerCard data={scoredData} />)

    // Classification conveyed as a text label (Req 19.2), before any detail.
    expect(screen.getByText(/likely inflated/i)).toBeInTheDocument()
    // Effective discount percentage shown alongside the classification (Req 2.4).
    expect(screen.getByText(/70% off/i)).toBeInTheDocument()
    // The real (paid) and shown (reference) prices are both present (Req 2.4).
    expect(screen.getByText('₹1,499')).toBeInTheDocument()
    expect(screen.getByText('₹4,999')).toBeInTheDocument()
  })

  it('exposes a text alternative that lists the contributions when expanded', async () => {
    const user = userEvent.setup()
    render(<DiscountCheckerCard data={scoredData} />)

    // Detail (the SHAP explanation) is hidden until the card is expanded (Req 8.3).
    expect(screen.queryByText(/Review volume/i)).not.toBeInTheDocument()

    // Expand via keyboard: the disclosure control is a focusable button (Req 8.3).
    await user.tab()
    expect(screen.getByRole('button', { name: /discount/i })).toHaveFocus()
    await user.keyboard('{Enter}')

    // Text alternative describes each contribution in words (Req 19.5 groundwork),
    // strongest-|impact| first, with its direction.
    expect(
      screen.getByText(/pushed toward inflated by 18\.5/i),
    ).toBeInTheDocument()
    expect(screen.getByText(/Review volume/i)).toBeInTheDocument()
    expect(screen.getByText(/pushed toward genuine by 4\.1/i)).toBeInTheDocument()

    // The dynamically-imported (mocked) Plotly chart mounts once expanded.
    const chart = await screen.findByTestId('plotly-mock')
    expect(chart).toBeInTheDocument()
    // Base value + 3 contributions + total = a single waterfall trace.
    expect(chart).toHaveAttribute('data-trace-count', '1')
  })

  it('renders the unavailable message when the module is contained/absent', () => {
    const message = 'Discount analysis is unavailable for this product.'
    render(<DiscountCheckerCard data={{ available: false, message }} />)

    expect(screen.getByText(message)).toBeInTheDocument()
    expect(screen.getByRole('status')).toBeInTheDocument()
  })

  it('shows price context but no score, plus the limitation message, when verification is limited', async () => {
    const user = userEvent.setup()
    const limited = {
      genuineness_score: null,
      classification: 'verification_limited',
      message:
        'Category price statistics are unavailable for this product; showing available price context only.',
      displayed_price: 1499.0,
      reference_price: 4999.0,
      effective_discount_pct: 70.0,
    }
    render(<DiscountCheckerCard data={limited} />)

    // Price context is still presented (Req 2.6) with a neutral, text-labelled badge.
    expect(screen.getByText(/verification limited/i)).toBeInTheDocument()
    expect(screen.getByText(/70% off/i)).toBeInTheDocument()
    // No genuineness score is claimed.
    expect(screen.queryByText(/% genuine/i)).not.toBeInTheDocument()

    // The limitation is disclosed in the detail region instead of a chart.
    await user.tab()
    await user.keyboard('{Enter}')
    expect(screen.getByText(limited.message)).toBeInTheDocument()
    expect(screen.queryByTestId('plotly-mock')).not.toBeInTheDocument()
  })
})
