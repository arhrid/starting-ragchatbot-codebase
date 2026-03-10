# Frontend Changes: Dark/Light Theme Toggle

## Summary
Added a toggle button that allows users to switch between dark and light themes with smooth transitions and localStorage persistence.

## Files Modified

### `frontend/index.html`
- Added a theme toggle button (`#themeToggle`) positioned before the main container
- Button contains two SVG icons: moon (dark mode) and sun (light mode)
- Includes `aria-label` and `title` for accessibility
- Bumped `style.css` version to `v=12` and `script.js` to `v=10`

### `frontend/style.css`
- Added `[data-theme="light"]` selector with a full set of light theme CSS variables
- Extracted hardcoded colors into CSS variables: `--code-bg`, `--source-link-color`, `--source-bg`, `--source-border`, `--source-hover-bg`, `--source-hover-border`, `--source-hover-accent`, `--source-hover-color`, `--link-color`, `--link-hover-color`, `--link-underline`, `--link-hover-underline`, `--error-bg`, `--error-color`, `--error-border`, `--success-bg`, `--success-color`, `--success-border`
- Updated all previously hardcoded color values (sources, links, code blocks, error/success messages) to use the new variables
- Added global theme transition (`background-color`, `color`, `border-color`, `box-shadow` with 0.3s ease)
- Added `.theme-toggle` button styles: fixed position top-right, circular, hover/focus states

### `frontend/script.js`
- Added `initTheme()` — reads saved theme from `localStorage` and applies `data-theme` attribute
- Added `toggleTheme()` — toggles between dark/light, saves preference to `localStorage`
- Added `updateThemeIcon()` — shows/hides sun/moon icons based on current theme
- Theme initializes before DOMContentLoaded to prevent flash of wrong theme
- Toggle button event listener registered in DOMContentLoaded

## Design Decisions
- Dark theme remains the default
- Theme preference persists via `localStorage` under the key `theme`
- Uses `data-theme` attribute on `<body>` for CSS variable switching
- All existing elements maintain visual hierarchy in both themes
- Toggle button is keyboard-navigable with focus ring styling
