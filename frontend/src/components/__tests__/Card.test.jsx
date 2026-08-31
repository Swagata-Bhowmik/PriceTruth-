import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

// framer-motion drives real animations (requestAnimationFrame + DOM
// measurement) that jsdom can't run. Replace it with synchronous passthrough
// elements so these behavioral tests are deterministic and noise-free; the
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
const { default: Card } = await import('../Card.jsx')

describe('Card (Req 8.3, 19.3, 19.4)', () => {
  it('always shows the title and the primary conclusion', () => {
    render(
      <Card title="Discount" primary={<span>Likely inflated</span>}>
        <p>Detail body</p>
      </Card>,
    )

    expect(screen.getByText('Discount')).toBeInTheDocument()
    expect(screen.getByText('Likely inflated')).toBeInTheDocument()
  })

  it('expands the detail on keyboard activation and reflects state via aria-expanded', async () => {
    const user = userEvent.setup()
    render(
      <Card title="Discount" primary={<span>Likely inflated</span>}>
        <p>Detail body</p>
      </Card>,
    )

    const toggle = screen.getByRole('button', { name: /discount/i })
    expect(toggle).toHaveAttribute('aria-expanded', 'false')
    expect(screen.queryByText('Detail body')).not.toBeInTheDocument()

    // Tab to the control: it is focusable, but focus alone does not expand.
    await user.tab()
    expect(toggle).toHaveFocus()
    expect(screen.queryByText('Detail body')).not.toBeInTheDocument()

    // Keyboard activation expands the detail region (Req 8.3).
    await user.keyboard('{Enter}')
    expect(screen.getByText('Detail body')).toBeInTheDocument()
    expect(toggle).toHaveAttribute('aria-expanded', 'true')

    // Activating again collapses it (toggle).
    await user.keyboard('{Enter}')
    expect(toggle).toHaveAttribute('aria-expanded', 'false')
  })

  it('reveals the detail on hover', async () => {
    const user = userEvent.setup()
    render(
      <Card title="Discount" primary={<span>Primary</span>}>
        <p>Detail body</p>
      </Card>,
    )

    expect(screen.queryByText('Detail body')).not.toBeInTheDocument()
    const card = screen.getByRole('button', { name: /discount/i }).closest('section')
    await user.hover(card)
    expect(screen.getByText('Detail body')).toBeInTheDocument()
  })

  it('exposes a focusable control with an explicit visible focus indicator', async () => {
    const user = userEvent.setup()
    render(
      <Card title="Discount" primary={<span>Primary</span>}>
        <p>Detail</p>
      </Card>,
    )

    const toggle = screen.getByRole('button', { name: /discount/i })
    await user.tab()
    expect(toggle).toHaveFocus()
    // The control carries an explicit focus-visible ring (Req 19.3).
    expect(toggle.className).toMatch(/focus-visible:ring/)
  })
})
