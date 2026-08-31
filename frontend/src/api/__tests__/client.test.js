import { describe, it, expect, afterEach } from 'vitest'
import {
  apiClient,
  searchProducts,
  checkDiscount,
  compareUnitPrice,
} from '../client.js'

// Preserve axios's default adapter so each test can install a stub without
// leaking into the next.
const originalAdapter = apiClient.defaults.adapter

afterEach(() => {
  apiClient.defaults.adapter = originalAdapter
})

describe('api client (Req 14.4, 15.3)', () => {
  it('unwraps a structured backend error into an Error carrying code/status/message/details', async () => {
    apiClient.defaults.adapter = async () => {
      const err = new Error('Request failed with status code 422')
      err.response = {
        status: 422,
        data: {
          error: {
            code: 'DISCOUNT_NOT_EVALUABLE',
            message: 'A discount cannot be evaluated because the reference price is invalid.',
            status: 422,
            details: { field: 'reference_price' },
          },
        },
      }
      throw err
    }

    await expect(checkDiscount({ displayed_price: 10 })).rejects.toMatchObject({
      isApiError: true,
      code: 'DISCOUNT_NOT_EVALUABLE',
      status: 422,
      message: 'A discount cannot be evaluated because the reference price is invalid.',
      details: { field: 'reference_price' },
    })
  })

  it('falls back to axios code/message when the payload is not structured', async () => {
    apiClient.defaults.adapter = async () => {
      const err = new Error('timeout of 5000ms exceeded')
      err.code = 'ECONNABORTED'
      throw err
    }

    await expect(searchProducts('milk')).rejects.toMatchObject({
      code: 'ECONNABORTED',
      message: 'timeout of 5000ms exceeded',
    })
  })

  it('sends the search query to GET /api/v1/search and returns the response body', async () => {
    let captured
    apiClient.defaults.adapter = async (config) => {
      captured = config
      return {
        data: { results: [{ product_id: 'p1' }] },
        status: 200,
        statusText: 'OK',
        headers: {},
        config,
      }
    }

    const data = await searchProducts('milk')
    expect(captured.method).toBe('get')
    expect(captured.url).toContain('/api/v1/search')
    expect(captured.params).toEqual({ q: 'milk' })
    expect(data).toEqual({ results: [{ product_id: 'p1' }] })
  })

  it('wraps variants under { variants } for the unit-price compare endpoint', async () => {
    let captured
    apiClient.defaults.adapter = async (config) => {
      captured = config
      return {
        data: { comparison: [], excluded: [] },
        status: 200,
        statusText: 'OK',
        headers: {},
        config,
      }
    }

    const variants = [{ label: 'S', price: 45, quantity: 100, unit: 'g' }]
    await compareUnitPrice(variants)
    expect(captured.method).toBe('post')
    expect(captured.url).toContain('/api/v1/unit-price/compare')
    expect(JSON.parse(captured.data)).toEqual({ variants })
  })
})
