# Image Similarity Analyzer

A deep learning-powered web application that compares two images and provides similarity scores, visual heatmaps, and AI-generated insights.

## Features

- **Deep Learning Similarity Analysis** - Uses ResNet50 pre-trained model for feature extraction
- **Visual Heatmaps** - Color-coded similarity maps showing matching regions
- **AI-Powered Insights** - Natural language explanations powered by Google Gemini AI
- **Modern UI** - Responsive React frontend with Tailwind CSS
- **Fast API Backend** - High-performance REST API built with FastAPI
- **Real-time Processing** - Get results in seconds

## Why ResNet50 Over CLIP?

This implementation uses **ResNet50** instead of CLIP for several advantages:

- **Faster Processing** - ResNet50 is lighter and processes images ~2-3x faster than CLIP
- **Lower Memory Footprint** - Requires less RAM, making it suitable for resource-constrained environments
- **Simpler Architecture** - Easier to understand, modify, and deploy
- **Task-Specific Optimization** - Better for pure visual similarity without text context
- **Wider Compatibility** - Works seamlessly across different deployment platforms

### Want CLIP Instead?

If you need semantic similarity with text understanding, check out the **`CLIP`**:

```bash
git checkout CLIP
```

The CLIP branch offers:
- Text-to-image similarity
- Better semantic understanding
- Multi-modal capabilities

## Requirements

- Python 3.8+
- Node.js 14+
- Google Gemini API key

## 🛠️ Installation

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

Create a `.env` file with your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

Run the server:
```bash
uvicorn main:app --reload
```

### Frontend Setup

```bash
cd frontend
npm install
npm start
```

## 📖 Usage

1. Open the web application at `http://localhost:3000`
2. Upload two images you want to compare
3. Click "Compare Images"
4. View similarity score, heatmap, and AI-generated insights

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

