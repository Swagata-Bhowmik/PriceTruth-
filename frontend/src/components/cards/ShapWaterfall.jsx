import { useEffect, useMemo, useState } from 'react'

// Status colors reused from the Tailwind tone tokens (Req 19.1 contrast):
// green = a contribution pushing the verdict toward genuine, red = toward
// inflated. Color is a redundant cue only — the card also provides a text
// breakdown, so meaning is never conveyed by color alone (Req 19.2, 19.5).
const GENUINE = '#15803d' // green-700
const INFLATED = '#b91c1c' // red-700
const NEUTRAL = '#6b7280' // gray-500
const CONNECTOR = '#d1d5db' // gray-300

/**
 * A horizontal SHAP "waterfall": the model's base (expected) value, then each
 * feature contribution as a relative step, ending at the final genuineness
 * score (Req 3.2, 3.3). `contributions` is expected already sorted by |impact|
 * so the chart and the card's text alternative stay in the same order.
 *
 * react-plotly.js pulls in the full plotly.js bundle (~3MB), so it is imported
 * DYNAMICALLY on mount (i.e. only once a user expands a card to view the
 * breakdown), keeping the initial dashboard bundle small (Req 8.3). Isolating
 * the dynamic import here also lets tests `vi.mock` 'react-plotly.js' so jsdom
 * never tries to render real Plotly.
 *
 * The rendered chart is decorative for assistive tech (`aria-hidden`) because
 * the card supplies an equivalent text breakdown next to it (Req 19.5).
 *
 * Props:
 *   - baseValue     : number, the SHAP base/expected value.
 *   - finalScore    : number, the reconciled final score shown as the total.
 *   - contributions : [{ feature, impact, direction }] already sorted.
 */
export default function ShapWaterfall({ baseValue, finalScore, contributions = [] }) {
  const [Plot, setPlot] = useState(null)
  const [loadFailed, setLoadFailed] = useState(false)

  useEffect(() => {
    let active = true
    import('react-plotly.js')
      .then((mod) => {
        if (active) setPlot(() => mod.default)
      })
      .catch(() => {
        if (active) setLoadFailed(true)
      })
    return () => {
      active = false
    }
  }, [])

  const { data, layout } = useMemo(() => {
    const labels = [
      'Base value',
      ...contributions.map((c) => c.feature),
      'Final score',
    ]
    const measure = [
      'absolute',
      ...contributions.map(() => 'relative'),
      'total',
    ]
    const values = [
      baseValue,
      ...contributions.map((c) => c.impact),
      finalScore,
    ]

    const trace = {
      type: 'waterfall',
      orientation: 'h',
      y: labels,
      x: values,
      measure,
      connector: { line: { color: CONNECTOR } },
      // Sign of a relative step == its direction (Property 7), so mapping the
      // increasing/decreasing colors gives green=toward_genuine, red=toward_inflated.
      increasing: { marker: { color: GENUINE } },
      decreasing: { marker: { color: INFLATED } },
      totals: { marker: { color: NEUTRAL } },
      hovertemplate: '%{y}: %{x:+.2f}<extra></extra>',
    }

    const chartLayout = {
      margin: { l: 8, r: 8, t: 8, b: 28 },
      height: 48 + labels.length * 34,
      showlegend: false,
      font: { size: 12 },
      xaxis: { title: { text: 'Contribution to genuineness score' }, zeroline: true },
      yaxis: { automargin: true },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
    }

    return { data: [trace], layout: chartLayout }
  }, [baseValue, finalScore, contributions])

  if (loadFailed) {
    return (
      <div role="status" className="py-4 text-xs text-gray-500">
        The chart could not be loaded; see the breakdown below.
      </div>
    )
  }

  if (!Plot) {
    return (
      <div role="status" className="py-4 text-xs text-gray-500">
        Loading chart…
      </div>
    )
  }

  return (
    // Decorative for AT: the sibling text breakdown is the accessible form.
    <div aria-hidden="true">
      <Plot
        data={data}
        layout={layout}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%' }}
        useResizeHandler
      />
    </div>
  )
}
