# Nuxt Minimal Starter

Look at the [Nuxt documentation](https://nuxt.com/docs/getting-started/introduction) to learn more.

## Setup

Make sure to install dependencies:

```bash
# npm
npm install

# pnpm
pnpm install

# yarn
yarn install

# bun
bun install
```

## Development Server

Start the development server on `http://localhost:3000`:

```bash
# npm
npm run dev

# pnpm
pnpm dev

# yarn
yarn dev

# bun
bun run dev
```

## Production

Build the application for production:

```bash
# npm
npm run build

# pnpm
pnpm build

# yarn
yarn build

# bun
bun run build
```

Locally preview production build:

```bash
# npm
npm run preview

# pnpm
pnpm preview

# yarn
yarn preview

# bun
bun run preview
```

Check out the [deployment documentation](https://nuxt.com/docs/getting-started/deployment) for more information.

## Component Documentation

### HeatmapViewer

Fixed left-overlay displaying segment importance across PDF documents. Supports visual scanning, click navigation, and drag-to-scroll.

**Features:**
- Canvas-based segment scoring visualization with brightness encoding
- Click navigation and drag-to-scroll viewport control
- Real-time image position indicators

**Props:**
```typescript
{
  highlights: Highlight[];          // Segment polygons stored by 0-indexed page number in polygons Record
  currentPage: number;              // Currently visible page (1-indexed)
  pageAspectRatio: number;          // Height/width ratio of PDF pages
  pageRefs: Element[];              // DOM references to PDF page elements
  editorStates: EditorImageState[]; // Image presence state per segment
}
```

**Events:**
```typescript
{
  navigate: (page: number, normalizedY: number) => void;  // Click-to-navigate (page is 0-indexed)
}
```

**Requirements:**
- Normalized coordinates [0,1] in highlight polygons
- Canvas 2D rendering context

**Performance:**
- Optimized for up to 500 segments and 100 pages
- Debounced rendering (200ms) for score updates

**Browser Support:**
- Chrome/Edge 90+, Firefox 88+, Safari 14+
