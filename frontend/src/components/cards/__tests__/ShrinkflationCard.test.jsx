import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

// recharts' ResponsiveContainer measures via ResizeObserver, which jsdom does
// not implement. Replace the recharts surface this card uses with inert
// passthrough/no-op elements so the presentational card renders deterministically
// without touching layout APIs. The card's own chart wrapper (role="img") and
// the text-alternative table/sources live OUTSIDE recharts, so they still render.
vi.mock('recharts', async () => {
  const React = await import('react')
  const Passthrough = ({ children }) => React.createElement('div', null, children)
  const Noop = () => null
  return {
    __esModule: true,
    ResponsiveContainer: Passthrough,
    LineChart: Passthrough,
    Line: Noop,
    XAxis: Noop,
    YAxis: Noop,
    CartesianGrid: Noop,
    Tooltip: Noop,
    Legend: Noop,
  }
})

// framer-motion (used by the shared Card) drives RAF-based animations jsdom
// can't run; swap it for synchronous passthroughs so expanded detail is present
// in the DOM immediately (mirrors the shared Card component test).
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

// Import after mocks are registered (vi.mock is hoisted).
const { default: ShrinkflationCard } = await import('../ShrinkflationCard.jsx')

// A representative "ok" payload matching the shrinkflation service shape:
// pack shrinks 100g -> 75g at a steady ₹100 price, so unit price rises ~33%.
const OK_DATA = {
  status: 'ok',
  points: [
    {
      observed_at: '2020-01-15',
      pack_quantity: 100,
      pack_unit: 'g',
      selling_price: 100,
      unit_price: 1.0,
      source_type: 'off',
      source_citation: null,
    },
    {
      observed_at: '2023-06-10',
      pack_quantity: 75,
      pack_unit: 'g',
      selling_price: 100,
      unit_price: 1.3333,
      source_type: 'cited_public_record',
      source_citation: 'Nielsen retail audit 2023',
    },
  ],
  total_change: {
    period_start: '2020-01-15',
    period_end: '2023-06-10',
    pack_quantity_pct: -25,
    unit_price_pct: 33.33,
  },
  message: null,
}

describe('ShrinkflationCard (Req 4.1, 4.4, 8.3, 19.4, 19.5)', () => {
  it('shows the primary total percentage changes as the conclusion (Req 19.4)', () => {
    render(<ShrinkflationCard data={OK_DATA} />)

    // Primary conclusion is visible without expanding.
    expect(screen.getByText('-25%')).toBeInTheDocument()
    expect(screen.getByText('+33%')).toBeInTheDocument()
    // Rounded for display: 33.33 -> +33 (raw value not shown verbatim).
    expect(screen.queryByText(/33\.33/)).not.toBeInTheDocument()
    // A text label accompanies the tone, not color alone.
    expect(screen.getByText('Shrinkflation detected')).toBeInTheDocument()
  })

  it('reveals source attribution for each point when expanded (Req 4.4, 8.3)', async () => {
    const user = userEvent.setup()
    render(<ShrinkflationCard data={OK_DATA} />)

    // Detail is hidden until the card is expanded.
    expect(screen.queryByText(/Nielsen retail audit 2023/)).not.toBeInTheDocument()

    // Expand via keyboard activation (Req 8.3): focus the disclosure control
    // then press Enter.
    await user.tab()
    await user.keyboard('{Enter}')

    // Source type labels + citation are attributed per point.
    expect(screen.getByText(/Open Food Facts \(crowd-sourced\)/)).toBeInTheDocument()
    expect(screen.getByText(/Cited public record/)).toBeInTheDocument()
    expect(screen.getByText(/Nielsen retail audit 2023/)).toBeInTheDocument()
  })

  it('provides a text alternative table for the chart timeline (Req 19.5)', async () => {
    const user = userEvent.setup()
    render(<ShrinkflationCard data={OK_DATA} />)

    await user.tab()
    await user.keyboard('{Enter}')

    // The chart itself carries a descriptive text alternative.
    expect(
      screen.getByRole('img', { name: /pack quantity and unit price over time/i }),
    ).toBeInTheDocument()

    // A data table exposes the same timeline without the chart.
    expect(screen.getByText('Pack-size timeline (text alternative)')).toBeInTheDocument()
    const table = screen.getByRole('table')
    expect(table).toBeInTheDocument()
    // Per-point values are present as text.
    expect(screen.getByText('100 g')).toBeInTheDocument()
    expect(screen.getByText('75 g')).toBeInTheDocument()
    // Unit price is computed/displayed per point.
    expect(screen.getByText(/₹1\.000\/g/)).toBeInTheDocument()
    expect(screen.getByText(/₹1\.333\/g/)).toBeInTheDocument()
  })

  it('summarizes a single recorded pack size without a percentage change', () => {
    const singlePoint = {
      status: 'ok',
      points: [
        {
          observed_at: '2022-03-01',
          pack_quantity: 200,
          pack_unit: 'ml',
          selling_price: 50,
          unit_price: 0.25,
          source_type: 'off',
          source_citation: null,
        },
      ],
      total_change: null,
      message: null,
    }
    render(<ShrinkflationCard data={singlePoint} />)

    expect(screen.getByText(/Single recorded pack size/)).toBeInTheDocument()
    expect(screen.getByText(/200 ml/)).toBeInTheDocument()
  })

  it('renders the unavailable message for an unavailable module (Req 8.5)', () => {
    const message = 'Pack-size history is unavailable for this product.'
    render(
      <ShrinkflationCard
        data={{ status: 'unavailable', points: [], total_change: null, message }}
      />,
    )

    expect(screen.getByText(message)).toBeInTheDocument()
    expect(screen.getByRole('status')).toBeInTheDocument()
    // No timeline chart/table is rendered in the unavailable state.
    expect(screen.queryByRole('table')).not.toBeInTheDocument()
  })

  it('supports the alternate { available: false, message } unavailable shape (Req 8.5)', () => {
    const message = 'No pack-size records found.'
    render(<ShrinkflationCard data={{ available: false, message }} />)

    expect(screen.getByText(message)).toBeInTheDocument()
    expect(screen.getByRole('status')).toBeInTheDocument()
  })
})
