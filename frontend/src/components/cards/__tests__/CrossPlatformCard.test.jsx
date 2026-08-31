import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

// framer-motion drives real animations (requestAnimationFrame + DOM
// measurement) that jsdom can't run. Replace it with synchronous passthrough
// elements so the shared Card's expand/collapse is deterministic here; the
// production Card still uses the real library.
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

// Import after the mock is registered (vi.mock is hoisted, so this is safe).
const { default: CrossPlatformCard } = await import('../CrossPlatformCard.jsx')

// Two platforms: the cheaper Flipkart listing carries a genuineness score and
// is the best deal; the Amazon listing has no score.
const multiPlatform = {
  product_id: 'amz_1',
  available: true,
  comparison_available: true,
  best_deal_platform: 'Flipkart',
  platforms: [
    {
      platform: 'Amazon',
      price: 1499,
      product_url: 'https://amazon.example/p/1',
    },
    {
      platform: 'Flipkart',
      price: 1299,
      product_url: 'https://flipkart.example/p/1',
      genuineness_score: 88,
      best_deal: true,
    },
  ],
  message: null,
}

const singlePlatform = {
  product_id: 'amz_2',
  available: true,
  comparison_available: false,
  best_deal_platform: null,
  platforms: [
    {
      platform: 'Amazon',
      price: 999,
      product_url: 'https://amazon.example/p/2',
    },
  ],
  message:
    'Price data is available on only one platform, so no cross-platform comparison is available.',
}

// The dashboard's contained slot for a product with no platform data.
const unavailable = {
  available: false,
  message: 'Cross-platform data is unavailable for this product.',
}

const expandCard = async (user) =>
  user.click(screen.getByRole('button', { name: /cross-platform/i }))

describe('CrossPlatformCard (Req 7.1-7.5, 8.3, 19.4)', () => {
  it('shows the best deal as the primary conclusion and links every platform', async () => {
    const user = userEvent.setup()
    render(<CrossPlatformCard data={multiPlatform} />)

    // PRIMARY conclusion, visible before any detail (Req 19.4): the best deal
    // is the cheapest platform (Req 7.2).
    const primary = screen.getByText(/best deal:/i).closest('p')
    expect(primary).toHaveTextContent(/Flipkart at .*1,299/)

    // Expand to reveal the per-platform detail (Req 8.3).
    await expandCard(user)

    // Every entry provides a product link that opens safely in a new tab
    // (Req 7.3), ordered cheapest-first.
    const links = screen.getAllByRole('link')
    expect(links).toHaveLength(2)
    links.forEach((link) => {
      expect(link).toHaveAttribute('target', '_blank')
      expect(link).toHaveAttribute('rel', 'noopener noreferrer')
    })
    expect(links[0]).toHaveAttribute('href', 'https://flipkart.example/p/1')
    expect(links[1]).toHaveAttribute('href', 'https://amazon.example/p/1')
  })

  it('shows a genuineness score only for the listing that has one', async () => {
    const user = userEvent.setup()
    render(<CrossPlatformCard data={multiPlatform} />)
    await expandCard(user)

    // Only the Flipkart listing has a score, so exactly one score is rendered
    // and it carries that value (Req 7.4).
    const scores = screen.getAllByText(/genuineness/i)
    expect(scores).toHaveLength(1)
    expect(scores[0]).toHaveTextContent('88')
  })

  it('shows the single price and the no-comparison message for one platform', () => {
    render(<CrossPlatformCard data={singlePlatform} />)

    // Single platform: its price plus the no-comparison message (Req 7.5), and
    // nothing marked as a best deal.
    expect(screen.getByText(/999/)).toBeInTheDocument()
    expect(screen.getByText(/only one platform/i)).toBeInTheDocument()
    expect(screen.queryByText(/best deal/i)).not.toBeInTheDocument()
  })

  it('renders the unavailable message when no platform data exists', () => {
    render(<CrossPlatformCard data={unavailable} />)

    // Req 7.6 / 8.5: the module's unavailable message is surfaced as a status.
    expect(screen.getByRole('status')).toBeInTheDocument()
    expect(
      screen.getByText(/cross-platform data is unavailable/i),
    ).toBeInTheDocument()
  })
})
