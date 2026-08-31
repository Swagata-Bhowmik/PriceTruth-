// Central runtime configuration for the frontend.
//
// API base URL: empty string by default so the Vite dev-server proxy
// (see vite.config.js) forwards `/api` and `/health` to the backend at
// http://localhost:8000. In production, set VITE_API_BASE_URL to the deployed
// backend origin (e.g. https://api.example.com) so requests target it directly.
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? ''
