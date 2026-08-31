import { describe, it, expect, vi } from 'vitest'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

// The shared Card (rendered by UnitPriceCard) drives its expand/collapse with
// framer-motion, which needs requestAnimationFrame + DOM measurement that jsdom
// can't run. Replace it with synchronous passthrough elements so the detail
// region appears deterministically on activation. Mirrors Card.test.jsx.
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

// Recharts renders an SVG chart via ResponsiveContainer, which relies on
// ResizeObserver + layout measurement unavailable in jsdom. Swap the pieces the
// card uses for minimal DOM stand-ins; the card's accessible content (table +
// text alternative) is independent of the chart internals.
vi.mock('recharts', async () => {
  const React = await import('react')
  const box = (testid) =>
    function MockRecharts({ children }) {
      return React.createElement('div', { 'data-testid': testid }, children)
    }
  return {
    __esModule: true,
    ResponsiveContainer: box('mock-responsive-container'),
    BarChart: box('mock-bar-chart'),
    Bar: box('mock-bar'),
    Cell: () => null,
    XAxis: () => null,
    YAxis: () => null,
    Tooltip: () => null,
    CartesianGrid: () => null,
  }
})

// Import after the mocks are registered (vi.mock is hoisted, so this is safe).
const { default: UnitPriceCard } = await import('../UnitPriceCard.jsx')

const moduleData = {
  standard_unit: 'g',
  comparison: [
    { label: 'Small', price: 45.0, quantity_std: 100, unit_price: 0.45 },
    { label: 'Family', price: 199.0, quantity_std: 1000, unit_price: 0.199, best_value: true },
  ],
  excluded: [{ label: 'Broken', reason: 'non_positive_quantity' }],
}

async function expandCard() {
  const user = userEvent.setup()
  await user.click(screen.getByRole('button', { name: /unit price/i }))
  return user
}

describe('UnitPriceCard (Req 5.2, 5.3, 8.3, 19.4, 19.5)', () => {
  it('shows the best-value variant as the primary conclusion, before any detail (Req 5.2, 19.4)', () => {
    render(<UnitPriceCard {...moduleData} />)

    // Primary conclusion is visible without expanding.
    expect(screen.getByText('Best value: Family at ₹0.199/g')).toBeInTheDocument()
    // Detail (table + chart) is not present until the card is expanded.
    expect(screen.queryByTestId('unit-price-table')).not.toBeInTheDocument()
    expect(screen.queryByTestId('mock-bar-chart')).not.toBeInTheDocument()
  })

  it('reveals a per-variant table with price, pack quantity and unit price on expand (Req 5.3, 8.3)', async () => {
    render(<UnitPriceCard {...moduleData} />)
    await expandCard()

    // The chart itself is rendered in the detail region.
    expect(screen.getByTestId('mock-bar-chart')).toBeInTheDocument()

    const table = screen.getByTestId('unit-price-table')
    // Column headers.
    expect(within(table).getByText('Variant')).toBeInTheDocument()
    expect(within(table).getByText('Price')).toBeInTheDocument()
    expect(within(table).getByText('Pack qty')).toBeInTheDocument()
    expect(within(table).getByText('Unit price')).toBeInTheDocument()

    // Small variant row.
    expect(within(table).getByText('Small')).toBeInTheDocument()
    expect(within(table).getByText('₹45.00')).toBeInTheDocument()
    expect(within(table).getByText('100 g')).toBeInTheDocument()
    expect(within(table).getByText('₹0.45/g')).toBeInTheDocument()

    // Family variant row (the best value).
    expect(within(table).getByText('Family')).toBeInTheDocument()
    expect(within(table).getByText('₹199.00')).toBeInTheDocument()
    expect(within(table).getByText('1000 g')).toBeInTheDocument()
    expect(within(table).getByText('₹0.199/g')).toBeInTheDocument()
    // Best-value marker is shown within the table (text + color, Req 5.2).
    expect(within(table).getByText(/best value/i)).toBeInTheDocument()
  })

  it('provides a text alternative for the chart carrying the same per-variant data (Req 19.5)', async () => {
    render(<UnitPriceCard {...moduleData} />)
    await expandCard()

    const textAlt = screen.getByTestId('unit-price-text-alt')
    expect(textAlt).toHaveTextContent(/Small at ₹0.45\/g/)
    expect(textAlt).toHaveTextContent(/Family at ₹0.199\/g \(best value\)/)
  })

  it('lists excluded variants together with their reason (Req 5.5)', async () => {
    render(<UnitPriceCard {...moduleData} />)
    await expandCard()

    const excluded = screen.getByTestId('unit-price-excluded')
    expect(within(excluded).getByText('Broken')).toBeInTheDocument()
    expect(within(excluded).getByText(/zero or negative/i)).toBeInTheDocument()
  })

  it('falls back to the lowest unit price when no variant is flagged best_value (Req 5.2)', () => {
    render(
      <UnitPriceCard
        standard_unit="ml"
        comparison={[
          { label: 'A', price: 30, quantity_std: 100, unit_price: 0.3 },
          { label: 'B', price: 50, quantity_std: 250, unit_price: 0.2 },
        ]}
      />,
    )

    expect(screen.getByText('Best value: B at ₹0.2/ml')).toBeInTheDocument()
  })

  it('renders the unavailable message and cannot expand when available is false (Req 8.5)', () => {
    render(
      <UnitPriceCard available={false} message="Unit-price data is unavailable for this product." />,
    )

    expect(screen.getByRole('status')).toHaveTextContent(
      'Unit-price data is unavailable for this product.',
    )
    // The card has no detail to reveal, so its disclosure control is disabled.
    expect(screen.getByRole('button', { name: /unit price/i })).toBeDisabled()
    expect(screen.queryByTestId('unit-price-table')).not.toBeInTheDocument()
  })

  it('renders an unavailable state when there are no comparison entries', () => {
    render(<UnitPriceCard standard_unit="g" comparison={[]} />)

    expect(screen.getByRole('status')).toBeInTheDocument()
  })
})
