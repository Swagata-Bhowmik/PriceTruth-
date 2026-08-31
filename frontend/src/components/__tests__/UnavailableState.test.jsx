import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import UnavailableState from '../UnavailableState.jsx'

describe('UnavailableState (Req 8.5)', () => {
  it('renders the provided unavailable message', () => {
    const message = 'Pack-size history is unavailable for this product.'
    render(<UnavailableState message={message} />)

    expect(screen.getByText(message)).toBeInTheDocument()
    expect(screen.getByRole('status')).toBeInTheDocument()
  })
})
