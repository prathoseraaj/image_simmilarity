import React, { useState } from 'react';
import axios from 'axios';

interface ComparisonResult {
  similarity_score: number;
  heatmap_image1: string;
  heatmap_image2: string;
  insights: string;
}

function App() {
  const [image1, setImage1] = useState<File | null>(null);
  const [image2, setImage2] = useState<File | null>(null);
  const [preview1, setPreview1] = useState<string>('');
  const [preview2, setPreview2] = useState<string>('');
  const [result, setResult] = useState<ComparisonResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');

  const handleImage1Change = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImage1(file);
      setPreview1(URL.createObjectURL(file));
      setResult(null);
      setError('');
    }
  };

  const handleImage2Change = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImage2(file);
      setPreview2(URL.createObjectURL(file));
      setResult(null);
      setError('');
    }
  };

  const handleCompare = async () => {
    if (!image1 || !image2) {
      setError('Please select both images');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    const formData = new FormData();
    formData.append('file1', image1);
    formData.append('file2', image2);

    try {
      const response = await axios.post<ComparisonResult>(
        'http://127.0.0.1:8000/compare/',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      setResult(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to compare images. Make sure the backend is running.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-linear-to-br from-blue-50 to-indigo-100 py-8 px-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            Image Similarity Analyzer
          </h1>
          <p className="text-gray-600">
            Upload two images to compare their similarity
          </p>
        </div>

        {/* Image Upload Section */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            {/* Image 1 Upload */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Image 1
              </label>
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center hover:border-blue-500 transition-colors">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImage1Change}
                  className="hidden"
                  id="image1-upload"
                />
                <label
                  htmlFor="image1-upload"
                  className="cursor-pointer block"
                >
                  {preview1 ? (
                    <img
                      src={preview1}
                      alt="Preview 1"
                      className="max-h-64 mx-auto rounded"
                    />
                  ) : (
                    <div className="py-12">
                      <svg
                        className="mx-auto h-12 w-12 text-gray-400"
                        stroke="currentColor"
                        fill="none"
                        viewBox="0 0 48 48"
                      >
                        <path
                          d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                          strokeWidth={2}
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </svg>
                      <p className="mt-2 text-sm text-gray-600">
                        Click to upload image
                      </p>
                    </div>
                  )}
                </label>
              </div>
            </div>

            {/* Image 2 Upload */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Image 2
              </label>
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center hover:border-blue-500 transition-colors">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImage2Change}
                  className="hidden"
                  id="image2-upload"
                />
                <label
                  htmlFor="image2-upload"
                  className="cursor-pointer block"
                >
                  {preview2 ? (
                    <img
                      src={preview2}
                      alt="Preview 2"
                      className="max-h-64 mx-auto rounded"
                    />
                  ) : (
                    <div className="py-12">
                      <svg
                        className="mx-auto h-12 w-12 text-gray-400"
                        stroke="currentColor"
                        fill="none"
                        viewBox="0 0 48 48"
                      >
                        <path
                          d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                          strokeWidth={2}
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </svg>
                      <p className="mt-2 text-sm text-gray-600">
                        Click to upload image
                      </p>
                    </div>
                  )}
                </label>
              </div>
            </div>
          </div>

          {/* Compare Button */}
          <div className="text-center">
            <button
              onClick={handleCompare}
              disabled={!image1 || !image2 || loading}
              className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-8 rounded-lg disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors shadow-md hover:shadow-lg"
            >
              {loading ? (
                <span className="flex items-center">
                  <svg
                    className="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    ></circle>
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    ></path>
                  </svg>
                  Comparing...
                </span>
              ) : (
                'Compare Images'
              )}
            </button>
          </div>

          {/* Error Message */}
          {error && (
            <div className="mt-4 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
              {error}
            </div>
          )}
        </div>

        {/* Results Section */}
        {result && (
          <div className="space-y-6">
            {/* Similarity Score */}
            <div className="bg-white rounded-lg shadow-lg p-8">
              <h2 className="text-xl font-semibold text-gray-800 mb-6 border-b pb-3">
                Similarity Analysis
              </h2>
              <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-6">
                <div className="flex-1">
                  <div className="mb-2">
                    <span className="text-sm font-medium text-gray-600 uppercase tracking-wide">
                      Similarity Score
                    </span>
                  </div>
                  <div className="text-6xl font-bold text-gray-900 mb-1">
                    {(result.similarity_score * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-gray-500 font-medium">
                    {result.similarity_score > 0.8
                      ? 'High Correlation'
                      : result.similarity_score > 0.5
                      ? 'Moderate Correlation'
                      : 'Low Correlation'}
                  </div>
                </div>
                <div className="flex-1">
                  <div className="space-y-3">
                    <div className="flex justify-between text-sm text-gray-600 mb-1">
                      <span>Match Confidence</span>
                      <span className="font-medium">{(result.similarity_score * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-sm h-2.5 overflow-hidden">
                      <div
                        className="bg-blue-600 h-full transition-all duration-700 ease-out"
                        style={{ width: `${result.similarity_score * 100}%` }}
                      ></div>
                    </div>
                    <div className="flex justify-between text-xs text-gray-400 mt-2">
                      <span>0%</span>
                      <span>50%</span>
                      <span>100%</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Heatmap */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">
                Similarity Heatmaps
              </h2>
              <p className="text-sm text-gray-600 mb-4 text-center">
                The heatmaps show areas of similarity between the two images (red = high similarity, blue = low similarity)
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Heatmap for Image 1 */}
                <div>
                  <h3 className="text-lg font-semibold text-gray-700 mb-3 text-center">
                    Image 1 - Similarity Map
                  </h3>
                  <div className="flex justify-center">
                    <img
                      src={`data:image/png;base64,${result.heatmap_image1}`}
                      alt="Similarity Heatmap - Image 1"
                      className="max-w-full rounded-lg shadow-md border-2 border-gray-200"
                    />
                  </div>
                </div>
                
                {/* Heatmap for Image 2 */}
                <div>
                  <h3 className="text-lg font-semibold text-gray-700 mb-3 text-center">
                    Image 2 - Similarity Map
                  </h3>
                  <div className="flex justify-center">
                    <img
                      src={`data:image/png;base64,${result.heatmap_image2}`}
                      alt="Similarity Heatmap - Image 2"
                      className="max-w-full rounded-lg shadow-md border-2 border-gray-200"
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Insights */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">
                Analysis Insights
              </h2>
              <div className="prose max-w-none">
                <p className="text-gray-700 leading-relaxed whitespace-pre-line">
                  {result.insights}
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
