/**
 * Tailwind CSS configuration.
 *
 * Responsive breakpoints (Requirements 14.1, 14.2, 14.3):
 *   - Mobile   (<= 480px)      : single-column layout.
 *   - Tablet   (481px - 1023px): adapted tablet layout, no horizontal scroll.
 *   - Desktop  (>= 1024px)     : full multi-column dashboard grid.
 *
 * Tailwind's default breakpoints cover these ranges without customization:
 *   sm = 640px, md = 768px, lg = 1024px, xl = 1280px, 2xl = 1536px.
 * Unprefixed (base) styles apply to mobile (<= 480px); `sm` and `md` cover the
 * tablet band (481-1023px); `lg` and above target desktop (>= 1024px). The
 * dashboard grid (task 18.2) uses `lg:` to switch to the multi-column layout.
 */

/**
 * Status color tokens (Requirement 19.1 — WCAG AA contrast).
 *
 * Each status "tone" (genuine / moderate / inflated) exposes four shades so
 * status can always be conveyed by TEXT + COLOR together (Req 19.2), never by
 * color alone:
 *   - DEFAULT : accent used as solid color / small strips. As *normal text on
 *               white* it meets WCAG AA (>= 4.5:1):
 *                 genuine  #15803d  ~= 5.0:1
 *                 moderate #b45309  ~= 5.0:1
 *                 inflated #b91c1c  ~= 6.5:1
 *   - fg      : dark text shade used on the light `bg` tint (and on white).
 *               Contrast against white is >= 7:1 and against its own `bg` tint
 *               is comfortably above 4.5:1, so badge text is always AA-compliant.
 *   - bg      : light tint used as a badge / pill background.
 *   - border  : mid tint used for subtle 1px borders around the tint.
 * Large text only needs 3:1, so these tokens satisfy both thresholds.
 */
const statusTones = {
  genuine: {
    DEFAULT: '#15803d', // green-700
    fg: '#166534', // green-800
    bg: '#dcfce7', // green-100
    border: '#86efac', // green-300
  },
  moderate: {
    DEFAULT: '#b45309', // amber-700
    fg: '#92400e', // amber-800
    bg: '#fef3c7', // amber-100
    border: '#fcd34d', // amber-300
  },
  inflated: {
    DEFAULT: '#b91c1c', // red-700
    fg: '#991b1b', // red-800
    bg: '#fee2e2', // red-100
    border: '#fca5a5', // red-300
  },
}

/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        ...statusTones,
        // Brand accent used for the visible keyboard focus ring (Req 19.3).
        // #4f46e5 (indigo-600) on white ~= 6.3:1.
        focusring: '#4f46e5',
      },
    },
  },
  plugins: [],
}
