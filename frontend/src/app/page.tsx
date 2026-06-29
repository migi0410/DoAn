"use client";

import React, { useState, useRef } from "react";
import axios from "axios";
import { UploadCloud, FileImage, Settings2, Play, FileJson, LayoutTemplate, Loader2, Sparkles, CheckCircle2 } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [baseline, setBaseline] = useState<string>("layoutlmv3_craft");
  const [preprocess, setPreprocess] = useState<boolean>(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [resultImage, setResultImage] = useState<string | null>(null);
  const [preprocessedImage, setPreprocessedImage] = useState<string | null>(null);
  const [showPreprocessed, setShowPreprocessed] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      // Reset result when new file is uploaded
      setResult(null);
      setResultImage(null);
      setPreprocessedImage(null);
      setShowPreprocessed(false);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const selectedFile = e.dataTransfer.files[0];
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setResultImage(null);
      setPreprocessedImage(null);
      setShowPreprocessed(false);
    }
  };

  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);
    setResultImage(null);
    setPreprocessedImage(null);
    setShowPreprocessed(false);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("baseline", baseline);
    formData.append("preprocess", preprocess ? "true" : "false");

    try {
      // Local FastAPI URL for now (Update to Render URL when deployed)
      const apiUrl = "http://localhost:8005/api/predict";
      const response = await axios.post(apiUrl, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      setResult(response.data.extraction);
      // Backend returns a relative image url like /api/image/...
      setResultImage(`http://localhost:8005${response.data.image_url}`);
      if (response.data.preprocessed_url) {
        setPreprocessedImage(`http://localhost:8005${response.data.preprocessed_url}`);
        setShowPreprocessed(true); // Auto-show preprocessed image if available
      }
    } catch (error: any) {
      console.error(error);
      alert("Đã xảy ra lỗi khi trích xuất: " + (error.response?.data?.error || error.message));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0A0A0A] text-slate-200 font-sans selection:bg-indigo-500/30">
      {/* Background Glow */}
      <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] rounded-full bg-indigo-600/20 blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] rounded-full bg-blue-600/20 blur-[120px]" />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-6 py-12">
        {/* Header */}
        <header className="mb-12 text-center">
          <motion.div 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 mb-6"
          >
            <Sparkles className="w-4 h-4 text-indigo-400" />
            <span className="text-sm font-medium tracking-wide text-slate-300">AVIR-KIE Thesis Project</span>
          </motion.div>
          <motion.h1 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-4xl md:text-6xl font-extrabold tracking-tight mb-4 bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent"
          >
            Hệ thống Trích xuất Hóa Đơn
          </motion.h1>
          <motion.p 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="text-slate-400 max-w-2xl mx-auto text-lg"
          >
            So sánh trực quan giữa các phương pháp tiếp cận Rule-based (OCR) và mô hình học sâu (LayoutLM) trong bài toán Key Information Extraction.
          </motion.p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Left Column: Input & Controls */}
          <div className="lg:col-span-4 space-y-6">
            <motion.div 
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
              className="bg-white/5 border border-white/10 backdrop-blur-xl rounded-2xl p-6 shadow-2xl"
            >
              <div className="flex items-center gap-3 mb-6">
                <div className="p-2 bg-indigo-500/20 rounded-lg">
                  <Settings2 className="w-5 h-5 text-indigo-400" />
                </div>
                <h2 className="text-xl font-semibold text-white">Bảng Điều Khiển</h2>
              </div>

              {/* Baseline Selector */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-slate-400 mb-2">Chọn Mô Hình Thử Nghiệm</label>
                <div className="relative">
                  <select
                    value={baseline}
                    onChange={(e) => setBaseline(e.target.value)}
                    className="w-full appearance-none bg-black/40 border border-white/10 rounded-xl px-4 py-3 text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 transition-all"
                  >
                    <optgroup label="✨ Mô hình Đề xuất">
                      <option value="layoutlmv3_craft">LayoutLMv3 + CRAFT + VietOCR</option>
                    </optgroup>
                    <optgroup label="🤖 Mô hình Baseline">
                      <option value="layoutlmv3">LayoutLMv3 + PaddleOCR</option>
                      <option value="phobert">PhoBERT</option>
                      <option value="layoutlmv1">LayoutLMv1</option>
                    </optgroup>
                    <optgroup label="⚙️ Phương pháp Truyền thống">
                      <option value="rule_paddle">PaddleOCR + Luật Regex</option>
                      <option value="rule_craft_vietocr">CRAFT + VietOCR + Luật Regex</option>
                    </optgroup>
                  </select>
                  <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-slate-500">
                    ▼
                  </div>
                </div>
              </div>

              {/* OpenCV Toggle */}
              <div className="mb-8 p-4 bg-indigo-500/10 border border-indigo-500/20 rounded-xl flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-medium text-indigo-300">OpenCV Scanner</h3>
                  <p className="text-xs text-slate-400 mt-1">Cắt viền, xoay phẳng & khử bóng nền</p>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input 
                    type="checkbox" 
                    className="sr-only peer" 
                    checked={preprocess}
                    onChange={(e) => setPreprocess(e.target.checked)}
                  />
                  <div className="w-11 h-6 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-indigo-500"></div>
                </label>
              </div>

              {/* File Uploader */}
              <div>
                <label className="block text-sm font-medium text-slate-400 mb-2">Tải lên Hóa đơn</label>
                <div 
                  onClick={() => fileInputRef.current?.click()}
                  onDragOver={handleDragOver}
                  onDrop={handleDrop}
                  className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200 
                    ${file ? 'border-indigo-500/50 bg-indigo-500/5' : 'border-white/10 hover:border-white/20 hover:bg-white/5'}`}
                >
                  <input 
                    type="file" 
                    className="hidden" 
                    ref={fileInputRef} 
                    accept="image/*"
                    onChange={handleFileChange}
                  />
                  {preview ? (
                    <div className="flex flex-col items-center">
                      <div className="w-12 h-12 bg-indigo-500/20 rounded-full flex items-center justify-center mb-3">
                        <CheckCircle2 className="w-6 h-6 text-indigo-400" />
                      </div>
                      <p className="text-sm font-medium text-slate-200 truncate max-w-[200px]">{file?.name}</p>
                      <p className="text-xs text-slate-500 mt-1">Nhấp để chọn ảnh khác</p>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center">
                      <div className="w-12 h-12 bg-white/5 rounded-full flex items-center justify-center mb-3 group-hover:bg-white/10 transition-colors">
                        <UploadCloud className="w-6 h-6 text-slate-400" />
                      </div>
                      <p className="text-sm font-medium text-slate-300">Kéo thả ảnh hoặc nhấp để tải lên</p>
                      <p className="text-xs text-slate-500 mt-1">Hỗ trợ JPG, PNG</p>
                    </div>
                  )}
                </div>
              </div>

              {/* Submit Button */}
              <button 
                onClick={handlePredict}
                disabled={!file || loading}
                className={`w-full mt-8 py-4 rounded-xl font-bold flex items-center justify-center gap-2 transition-all duration-300 shadow-lg
                  ${!file || loading 
                    ? 'bg-white/5 text-slate-500 cursor-not-allowed border border-white/5' 
                    : 'bg-gradient-to-r from-indigo-500 to-blue-600 text-white hover:shadow-indigo-500/25 hover:-translate-y-0.5 border border-transparent'
                  }`}
              >
                {loading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Đang xử lý AI...
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    Chạy Trích Xuất KIE
                  </>
                )}
              </button>
            </motion.div>
          </div>

          {/* Right Column: Viewer */}
          <div className="lg:col-span-8">
            <motion.div 
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
              className="bg-white/5 border border-white/10 backdrop-blur-xl rounded-2xl shadow-2xl overflow-hidden h-full min-h-[600px] flex flex-col"
            >
              
              {!result && !preview && !loading && (
                <div className="flex-1 flex flex-col items-center justify-center text-slate-500 p-12 text-center">
                  <LayoutTemplate className="w-16 h-16 mb-4 opacity-20" />
                  <h3 className="text-xl font-medium text-slate-300 mb-2">Chưa có dữ liệu hiển thị</h3>
                  <p>Vui lòng tải lên một ảnh hóa đơn và nhấp "Chạy Trích Xuất" để xem kết quả so sánh.</p>
                </div>
              )}

              {loading && (
                <div className="flex-1 flex flex-col items-center justify-center">
                  <Loader2 className="w-12 h-12 text-indigo-500 animate-spin mb-4" />
                  <p className="text-lg font-medium text-slate-300 animate-pulse">Đang phân tích hóa đơn...</p>
                </div>
              )}

              <AnimatePresence>
                {(result || (preview && !loading && !result)) && (
                  <motion.div 
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="grid grid-cols-1 md:grid-cols-2 h-full divide-y md:divide-y-0 md:divide-x divide-white/10"
                  >
                    
                    {/* Image Panel */}
                    <div className="p-6 bg-black/20 flex flex-col">
                      <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-2 text-slate-300">
                          <FileImage className="w-5 h-5 text-blue-400" />
                          <span className="font-medium">Ảnh Hóa Đơn</span>
                        </div>
                        {preprocessedImage && (
                          <div className="flex items-center gap-2">
                            <span className="text-xs text-slate-400">Kết quả (Có Box)</span>
                            <label className="relative inline-flex items-center cursor-pointer">
                              <input 
                                type="checkbox" 
                                className="sr-only peer" 
                                checked={showPreprocessed}
                                onChange={(e) => setShowPreprocessed(e.target.checked)}
                              />
                              <div className="w-9 h-5 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-indigo-500"></div>
                            </label>
                            <span className="text-xs text-indigo-400 font-medium">Bản làm sạch (OpenCV)</span>
                          </div>
                        )}
                      </div>
                      <div className="flex-1 rounded-xl overflow-hidden bg-black/40 border border-white/5 flex items-center justify-center p-2 relative">
                        {showPreprocessed && preprocessedImage ? (
                          <img src={preprocessedImage} alt="Preprocessed" className="max-h-[600px] object-contain rounded-lg shadow-2xl" />
                        ) : resultImage ? (
                          <img src={resultImage} alt="Result" className="max-h-[600px] object-contain rounded-lg shadow-2xl" />
                        ) : preview ? (
                          <img src={preview} alt="Preview" className="max-h-[600px] object-contain rounded-lg opacity-70" />
                        ) : null}
                      </div>
                    </div>

                    {/* Entities Panel */}
                    <div className="p-6 flex flex-col border-l border-white/10">
                      <div className="flex items-center gap-2 mb-4 text-slate-300">
                        <FileJson className="w-5 h-5 text-emerald-400" />
                        <span className="font-medium">Kết quả Trích Xuất (Entities)</span>
                      </div>
                      <div className="flex-1 overflow-auto pr-2">
                        {result ? (
                          <div className="space-y-3">
                            {Object.entries(result).map(([key, val]) => (
                              <div key={key} className="bg-black/40 border border-white/5 rounded-lg p-3 flex flex-col">
                                <span className="text-xs font-semibold text-indigo-400 mb-1">{key}</span>
                                {val === null || val === "" ? (
                                  <span className="text-slate-600 italic text-sm">Không tìm thấy</span>
                                ) : (
                                  <span className="text-slate-200 text-sm font-medium break-words">{String(val)}</span>
                                )}
                              </div>
                            ))}
                          </div>
                        ) : (
                          <div className="h-full flex items-center justify-center text-slate-600">
                            Chưa có dữ liệu trích xuất
                          </div>
                        )}
                      </div>
                    </div>

                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          </div>

        </div>
      </div>
    </div>
  );
}
