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

/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {},
  },
  plugins: [],
}
