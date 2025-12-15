# Caramella RAG Frontend

Modern React frontend for the GPU-accelerated RAG system.

## Setup

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Development

The frontend runs on `http://localhost:3000` and proxies API requests to the FastAPI backend at `http://localhost:8000`.

## Features

- 🚀 Real-time chat interface
- ⚡ Performance metrics display
- 📚 Source document viewer
- 📊 System statistics
- 🎨 Modern, responsive UI
- 🔄 Live health monitoring

## Configuration

Create a `.env` file (copy from `.env.example`):

```
VITE_API_URL=http://localhost:8000
```

## Tech Stack

- React 18
- Vite
- Axios
- CSS3 (no framework needed for this simple UI)
