import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import LoadingSkeleton from '../LoadingSkeleton.jsx'

describe('LoadingSkeleton (Req 8.4)', () => {
  it('renders an accessible loading status while a module is pending', () => {
    render(<LoadingSkeleton lines={2} />)

    const status = screen.getByRole('status')
    expect(status).toBeInTheDocument()
    expect(status).toHaveAttribute('aria-busy', 'true')
  })
})
