"use client";

import Image from "next/image";
import { useState, useRef, useEffect } from "react";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    return () => {
      if (preview) {
        URL.revokeObjectURL(preview);
      }
    };
  }, [preview]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleSubmit = async () => {
    if (!file) return;
    
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/correct", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Error connecting to backend:", error);
      alert("Failed to connect to the correction backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-8 pb-20 sm:p-20 font-[family-name:var(--font-geist-sans)]">
      <main className="flex flex-col gap-8 items-center sm:items-start max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold tracking-tight mb-4">AutoHDR Lens Correction</h1>
        
        <p className="text-lg text-gray-600 dark:text-gray-300 mb-8">
          Upload real estate photography to automatically correct barrel and pincushion distortion.
        </p>

        <div className="w-full grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div className="w-full flex flex-col gap-4 border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-xl p-8 items-center justify-center min-h-[300px] bg-gray-50 dark:bg-gray-900/50 transition-colors hover:border-gray-400 dark:hover:border-gray-500">
            <input 
              type="file" 
              accept="image/png, image/jpeg, image/jpg" 
              className="hidden" 
              ref={fileInputRef}
              onChange={handleFileChange}
            />
            
            {preview ? (
              <div className="w-full h-full flex flex-col items-center justify-center gap-4">
                <div className="relative w-full aspect-video rounded-lg overflow-hidden border border-gray-200 dark:border-gray-800">
                  <Image src={preview} alt="Preview" fill className="object-cover" />
                </div>
                <div className="flex gap-2 w-full">
                  <button 
                    onClick={handleUploadClick}
                    className="flex-1 px-4 py-2 border border-gray-300 dark:border-gray-700 rounded-full font-medium text-sm hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                  >
                    Change Image
                  </button>
                  <button 
                    onClick={handleSubmit}
                    disabled={loading}
                    className="flex-1 px-4 py-2 bg-black dark:bg-white text-white dark:text-black rounded-full font-medium text-sm hover:bg-gray-800 dark:hover:bg-gray-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {loading ? "Processing..." : "Correct Distortion"}
                  </button>
                </div>
              </div>
            ) : (
              <>
                <svg className="w-12 h-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                </svg>
                <div className="text-center">
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Click to upload or drag and drop</p>
                  <p className="text-xs text-gray-500">PNG, JPG, JPEG (max 10MB)</p>
                </div>
                <button 
                  onClick={handleUploadClick}
                  className="mt-4 px-6 py-2 bg-black dark:bg-white text-white dark:text-black rounded-full text-sm font-medium hover:bg-gray-800 dark:hover:bg-gray-200 transition-colors"
                >
                  Select Image
                </button>
              </>
            )}
          </div>

          {/* Results/Comparison Section */}
          <div className="w-full flex flex-col gap-4 border border-gray-200 dark:border-gray-800 rounded-xl p-8 min-h-[300px] bg-white dark:bg-black items-center justify-center shadow-sm">
            {!result ? (
              <p className="text-gray-500 text-sm text-center">
                {loading ? "Correcting lens distortion..." : "Upload an image and run correction to see the result and evaluation metrics here."}
              </p>
            ) : (
              <div className="w-full flex flex-col gap-4">
                <h3 className="font-semibold text-lg border-b pb-2">Correction Complete</h3>
                <div className="bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400 p-3 rounded-lg text-sm mb-2">
                  Status: {result.status}
                </div>
                
                <h4 className="font-medium text-sm mt-2 text-gray-500 uppercase tracking-wider">Image Metadata</h4>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div className="p-2 bg-gray-50 dark:bg-gray-900 rounded">Width: {result.width}px</div>
                  <div className="p-2 bg-gray-50 dark:bg-gray-900 rounded">Height: {result.height}px</div>
                </div>

                <h4 className="font-medium text-sm mt-4 text-gray-500 uppercase tracking-wider">Evaluation Metrics</h4>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-sm">
                  {result.metrics && Object.entries(result.metrics).map(([key, value]) => (
                    <div key={key} className="p-2 bg-gray-50 dark:bg-gray-900 rounded flex justify-between">
                      <span className="capitalize">{key.replace('_', ' ')}:</span>
                      <span className="font-mono">{Number(value).toFixed(2)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
