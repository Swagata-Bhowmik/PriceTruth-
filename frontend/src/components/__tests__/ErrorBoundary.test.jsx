import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import ErrorBoundary from '../ErrorBoundary.jsx'

// A child that throws during render, used to trip the boundary.
function Bomb() {
  throw new Error('boom')
}

describe('ErrorBoundary (Req 8.5, 15.1)', () => {
  afterEach(() => vi.restoreAllMocks())

  it('contains a throwing child and renders the fallback instead of propagating', () => {
    // React logs caught render errors to console.error; silence for a clean run.
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {})

    expect(() =>
      render(
        <ErrorBoundary>
          <Bomb />
        </ErrorBoundary>,
      ),
    ).not.toThrow()

    expect(screen.getByRole('alert')).toBeInTheDocument()
    expect(screen.getByText(/something went wrong/i)).toBeInTheDocument()
    spy.mockRestore()
  })

  it('renders its children unchanged when nothing throws', () => {
    render(
      <ErrorBoundary>
        <p>Healthy child</p>
      </ErrorBoundary>,
    )
    expect(screen.getByText('Healthy child')).toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('renders a custom fallback when one is provided', () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {})
    render(
      <ErrorBoundary fallback={<div>Custom fallback</div>}>
        <Bomb />
      </ErrorBoundary>,
    )
    expect(screen.getByText('Custom fallback')).toBeInTheDocument()
    spy.mockRestore()
  })
})
