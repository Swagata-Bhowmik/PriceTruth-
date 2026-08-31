import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'

// A tiny component to prove the jsdom + React Testing Library pipeline works.
function Hello() {
  return <h1>Price Truth</h1>
}

describe('test infrastructure smoke test', () => {
  it('runs the Vitest environment', () => {
    expect(true).toBe(true)
  })

  it('renders a component with jsdom + React Testing Library', () => {
    render(<Hello />)
    expect(
      screen.getByRole('heading', { name: 'Price Truth' }),
    ).toBeInTheDocument()
  })
})
