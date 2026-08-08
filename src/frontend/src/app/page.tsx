"use client";

import React, { useState, useRef } from "react";
import { Navbar } from "@/components/layout/Navbar";
import { useConfig } from "@/context/ConfigContext";
import { UploadCloud, Image as ImageIcon, Send, Sparkles, FileJson, Loader2, Bot, Layers, CheckCircle2, Download, Table, Trash2, Play } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import axios from "axios";

export default function Home() {
  const { apiUrl } = useConfig();
  
  // App State
  const [appMode, setAppMode] = useState<"single" | "batch">("single");

  // ================= SINGLE MODE STATE =================
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [baseline, setBaseline] = useState("layoutlmv3");
  const [preprocess, setPreprocess] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [result, setResult] = useState<any>(null);

  const [chatModel, setChatModel] = useState("gemini");
  const [question, setQuestion] = useState("");
  const [chatHistory, setChatHistory] = useState<{ role: "user" | "bot", content: string }[]>([]);
  const [isChatting, setIsChatting] = useState(false);

  // ================= BATCH MODE STATE =================
  const [batchFiles, setBatchFiles] = useState<File[]>([]);
  const [batchResults, setBatchResults] = useState<any[]>([]);
  const [isBatchProcessing, setIsBatchProcessing] = useState(false);
  const [batchProgress, setBatchProgress] = useState(0);
  const batchInputRef = useRef<HTMLInputElement>(null);

  // ================= HANDLERS (SINGLE) =================
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setChatHistory([]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => e.preventDefault();
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const selectedFile = e.dataTransfer.files[0];
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setChatHistory([]);
    }
  };

  const runPrediction = async () => {
    if (!file) return;
    setIsPredicting(true);
    setResult(null);
    
    const formData = new FormData();
    formData.append("file", file);
    formData.append("baseline", baseline);
    formData.append("preprocess", preprocess ? "true" : "false");

    try {
      const res = await axios.post(`${apiUrl}/api/predict`, formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });
      if (res.data.success || res.data.extraction) {
        setResult(res.data);
      } else {
        alert("Prediction failed.");
      }
    } catch (err) {
      console.error(err);
      alert("Error connecting to backend API.");
    } finally {
      setIsPredicting(false);
    }
  };

  const handleChat = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file || !question.trim() || isChatting) return;

    const q = question;
    setQuestion("");
    setChatHistory(prev => [...prev, { role: "user", content: q }]);
    setIsChatting(true);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("model", chatModel);
    formData.append("question", q);

    try {
      const res = await axios.post(`${apiUrl}/api/chat`, formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });
      
      if (res.data.success || res.data.answer) {
        setChatHistory(prev => [...prev, { role: "bot", content: res.data.answer }]);
      } else {
        setChatHistory(prev => [...prev, { role: "bot", content: "Sorry, an error occurred." }]);
      }
    } catch (err) {
      setChatHistory(prev => [...prev, { role: "bot", content: "Error connecting to backend API." }]);
    } finally {
      setIsChatting(false);
    }
  };

  // ================= HANDLERS (BATCH) =================
  const handleBatchFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setBatchFiles(prev => [...prev, ...Array.from(e.target.files!)]);
    }
  };
  
  const handleBatchDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setBatchFiles(prev => [...prev, ...Array.from(e.dataTransfer.files!)]);
    }
  };

  const clearBatchFiles = () => {
    setBatchFiles([]);
    setBatchResults([]);
    setBatchProgress(0);
  };

  const runBatchProcessing = async () => {
    if (batchFiles.length === 0) return;
    setIsBatchProcessing(true);
    setBatchResults([]);
    setBatchProgress(0);

    const results = [];
    for (let i = 0; i < batchFiles.length; i++) {
      const f = batchFiles[i];
      const formData = new FormData();
      formData.append("file", f);
      formData.append("baseline", baseline);
      formData.append("preprocess", preprocess ? "true" : "false");

      try {
        const res = await axios.post(`${apiUrl}/api/predict`, formData);
        const data = res.data.extraction || res.data;
        results.push({
          fileName: f.name,
          seller: data.SELLER || data.seller || "",
          address: data.ADDRESS || data.address || "",
          timestamp: data.TIMESTAMP || data.timestamp || "",
          totalCost: data.TOTAL_COST || data.total_cost || "",
          items: JSON.stringify(data.ITEM_NAME || data.items || []),
          status: "Success"
        });
      } catch (err) {
        results.push({
          fileName: f.name,
          seller: "", address: "", timestamp: "", totalCost: "", items: "",
          status: "Error"
        });
      }
      setBatchResults([...results]);
      setBatchProgress(Math.round(((i + 1) / batchFiles.length) * 100));
    }
    setIsBatchProcessing(false);
  };

  const exportCSV = () => {
    if (batchResults.length === 0) return;
    const headers = ["File Name", "Seller", "Address", "Timestamp", "Total Cost", "Items", "Status"];
    const csvRows = [headers.join(",")];

    for (const row of batchResults) {
      const values = [
        `"${row.fileName.replace(/"/g, '""')}"`,
        `"${String(row.seller).replace(/"/g, '""')}"`,
        `"${String(row.address).replace(/"/g, '""')}"`,
        `"${String(row.timestamp).replace(/"/g, '""')}"`,
        `"${String(row.totalCost).replace(/"/g, '""')}"`,
        `"${String(row.items).replace(/"/g, '""')}"`,
        `"${row.status}"`
      ];
      csvRows.push(values.join(","));
    }
    
    const blob = new Blob(["\uFEFF" + csvRows.join("\n")], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", `avir_kie_batch_export_${new Date().getTime()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };


  return (
    <div className="flex-1 flex flex-col h-full bg-slate-50 bg-[url('/grid.svg')] bg-center overflow-x-hidden">
      <Navbar />
      
      <main className="flex-1 max-w-[1600px] w-full mx-auto p-4 sm:p-6 lg:p-8 flex flex-col gap-6">
        
        {/* TOP CONTROLS & TABS */}
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 bg-white border border-slate-200 p-4 rounded-2xl shadow-sm">
          <div className="flex bg-slate-100 p-1 rounded-xl border border-slate-200">
            <button 
              onClick={() => setAppMode("single")}
              className={`px-6 py-2 rounded-lg text-sm font-bold transition-all ${appMode === 'single' ? 'bg-white text-blue-600 shadow-sm border border-slate-200/50' : 'text-slate-500 hover:text-slate-700'}`}
            >
              Single File Mode
            </button>
            <button 
              onClick={() => setAppMode("batch")}
              className={`px-6 py-2 rounded-lg text-sm font-bold transition-all ${appMode === 'batch' ? 'bg-white text-blue-600 shadow-sm border border-slate-200/50' : 'text-slate-500 hover:text-slate-700'}`}
            >
              Batch Processing Mode
            </button>
          </div>

          <div className="flex items-center gap-4 px-2">
            <div className="flex items-center gap-2">
              <label className="text-xs font-bold text-slate-500 uppercase tracking-wider">Model:</label>
              <select 
                value={baseline} 
                onChange={e => setBaseline(e.target.value)}
                className="bg-slate-50 border border-slate-200 rounded-lg px-3 py-1.5 text-sm text-slate-900 font-medium focus:outline-none focus:border-blue-500"
              >
                <option value="layoutlmv3_craft">LayoutLMv3 + CRAFT</option>
                <option value="layoutlmv3">LayoutLMv3</option>
                <option value="phobert">PhoBERT</option>
                <option value="qwen2_vl">Qwen2-VL</option>
                <option value="minicpm_v">MiniCPM-V</option>
              </select>
            </div>
            <label className="flex items-center gap-2 cursor-pointer">
              <input 
                type="checkbox" 
                checked={preprocess} 
                onChange={e => setPreprocess(e.target.checked)}
                className="rounded border-slate-300 text-blue-600 focus:ring-blue-500 w-4 h-4" 
              />
              <span className="text-sm font-bold text-slate-600">Auto Enhance</span>
            </label>
          </div>
        </div>

        {/* ================= SINGLE MODE UI ================= */}
        {appMode === "single" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            {/* LEFT COLUMN - CONTROLS & CHAT */}
            <div className="lg:col-span-5 flex flex-col gap-6">
              {/* UPLOAD SECTION */}
              <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xl shadow-slate-200/50">
                <h2 className="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
                  <ImageIcon className="w-5 h-5 text-blue-500" /> Document Input
                </h2>
                
                <div 
                  onDragOver={handleDragOver}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                  className={`border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer transition-all ${
                    preview ? "border-slate-300 bg-slate-50" : "border-blue-300 bg-blue-50 hover:bg-blue-100"
                  }`}
                >
                  <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" accept="image/*" />
                  {preview ? (
                    <div className="flex flex-col items-center gap-3 text-green-600">
                      <CheckCircle2 className="w-8 h-8" />
                      <span className="font-semibold text-sm text-center line-clamp-1">{file?.name}</span>
                      <span className="text-xs text-slate-500">Click to change document</span>
                    </div>
                  ) : (
                    <>
                      <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mb-3 shadow-sm">
                        <UploadCloud className="w-6 h-6 text-blue-600" />
                      </div>
                      <p className="text-sm font-semibold text-slate-700">Click or drag & drop</p>
                      <p className="text-xs text-slate-500 mt-1">SVG, PNG, JPG or PDF</p>
                    </>
                  )}
                </div>

                <div className="mt-6">
                  <button
                    onClick={runPrediction}
                    disabled={!file || isPredicting}
                    className="w-full relative group overflow-hidden rounded-xl p-[1px] shadow-md disabled:opacity-50"
                  >
                    <span className="absolute inset-0 bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500 rounded-xl opacity-90 group-hover:opacity-100 transition-opacity duration-300" />
                    <div className="relative flex items-center justify-center gap-2 bg-white px-4 py-3 rounded-xl transition-all group-hover:bg-opacity-0">
                      {isPredicting ? (
                        <Loader2 className="w-5 h-5 animate-spin text-blue-600 group-hover:text-white" />
                      ) : (
                        <Sparkles className="w-5 h-5 text-blue-600 group-hover:text-white" />
                      )}
                      <span className="font-bold text-slate-800 group-hover:text-white transition-colors">
                        {isPredicting ? "Processing Document..." : "Run Layout Analysis"}
                      </span>
                    </div>
                  </button>
                </div>
              </div>

              {/* CHAT SECTION */}
              <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xl shadow-slate-200/50 flex-1 flex flex-col min-h-[350px]">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-lg font-bold text-slate-800 flex items-center gap-2">
                    <Bot className="w-5 h-5 text-indigo-500" /> Document VLM Q&A
                  </h2>
                  <select 
                    value={chatModel} 
                    onChange={e => setChatModel(e.target.value)}
                    className="bg-slate-50 border border-slate-200 rounded-lg px-2 py-1 text-xs font-semibold text-slate-700 focus:outline-none focus:border-indigo-500 shadow-inner"
                  >
                    <option value="gemini">Gemini Pro Vision</option>
                    <option value="qwen2_vl">Qwen2-VL</option>
                    <option value="minicpm_v">MiniCPM-V</option>
                  </select>
                </div>

                <div className="flex-1 bg-slate-50 border border-slate-200 rounded-xl p-4 overflow-y-auto mb-4 space-y-4 max-h-[300px] shadow-inner">
                  {chatHistory.length === 0 ? (
                    <div className="h-full flex flex-col items-center justify-center text-slate-400">
                      <Bot className="w-8 h-8 mb-2 opacity-30" />
                      <p className="text-sm font-medium">Ask any question about the document.</p>
                    </div>
                  ) : (
                    chatHistory.map((msg, i) => (
                      <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                        <div className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-sm font-medium shadow-sm ${
                          msg.role === "user" 
                            ? "bg-blue-600 text-white rounded-br-none" 
                            : "bg-white text-slate-800 border border-slate-200 rounded-bl-none"
                        }`}>
                          {msg.content}
                        </div>
                      </div>
                    ))
                  )}
                  {isChatting && (
                    <div className="flex justify-start">
                      <div className="bg-white border border-slate-200 rounded-2xl rounded-bl-none px-4 py-3 flex items-center gap-2 shadow-sm">
                        <div className="w-2 h-2 bg-indigo-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                        <div className="w-2 h-2 bg-indigo-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                        <div className="w-2 h-2 bg-indigo-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                      </div>
                    </div>
                  )}
                </div>

                <form onSubmit={handleChat} className="relative">
                  <input 
                    type="text" 
                    value={question}
                    onChange={e => setQuestion(e.target.value)}
                    placeholder="Extract the total amount..."
                    disabled={!file || isChatting}
                    className="w-full bg-white border border-slate-200 shadow-inner rounded-xl pl-4 pr-12 py-3 text-sm font-medium text-slate-900 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 disabled:opacity-50"
                  />
                  <button 
                    type="submit"
                    disabled={!file || !question.trim() || isChatting}
                    className="absolute right-2 top-2 p-1.5 bg-indigo-600 text-white rounded-lg shadow-sm hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:hover:bg-indigo-600"
                  >
                    <Send className="w-4 h-4" />
                  </button>
                </form>
              </div>
            </div>

            {/* RIGHT COLUMN - RESULTS */}
            <div className="lg:col-span-7 flex flex-col">
              <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xl shadow-slate-200/50 h-full flex flex-col">
                <div className="flex items-center gap-2 mb-4 pb-4 border-b border-slate-100">
                  <Layers className="w-5 h-5 text-purple-600" />
                  <h2 className="text-lg font-bold text-slate-800">Analysis Results</h2>
                </div>
                
                {!preview ? (
                  <div className="flex-1 flex flex-col items-center justify-center text-slate-400 border-2 border-dashed border-slate-200 rounded-xl bg-slate-50">
                    <FileJson className="w-12 h-12 mb-3 opacity-30" />
                    <p className="font-medium">Upload a document and run analysis to see results here.</p>
                  </div>
                ) : (
                  <div className="flex-1 grid grid-cols-1 xl:grid-cols-2 gap-6 min-h-0">
                    <div className="flex flex-col h-full bg-slate-50 rounded-xl border border-slate-200 shadow-inner overflow-hidden">
                      <div className="bg-white px-3 py-2 border-b border-slate-200 text-xs font-bold text-slate-500 flex justify-between items-center uppercase tracking-wider">
                        <span>Document View</span>
                        {result?.preprocessed_url && <span className="text-green-600 bg-green-50 px-2 py-0.5 rounded-full border border-green-200">Preprocessed</span>}
                      </div>
                      <div className="flex-1 p-2 overflow-auto flex items-center justify-center">
                        <img 
                          src={result?.image_url ? (result.image_url.startsWith('http') ? result.image_url : `${apiUrl}${result.image_url}`) : preview} 
                          alt="Document" 
                          className="max-w-full max-h-[600px] object-contain rounded shadow-sm border border-slate-200 bg-white"
                        />
                      </div>
                    </div>

                    <div className="flex flex-col h-full bg-slate-50 rounded-xl border border-slate-200 shadow-inner overflow-hidden">
                      <div className="bg-white px-3 py-2 border-b border-slate-200 text-xs font-bold text-slate-500 uppercase tracking-wider">
                        Extracted JSON
                      </div>
                      <div className="flex-1 p-4 overflow-auto">
                        {isPredicting ? (
                          <div className="h-full flex flex-col items-center justify-center space-y-3">
                            <Loader2 className="w-8 h-8 text-blue-600 animate-spin" />
                            <p className="text-sm font-semibold text-slate-500 animate-pulse">Running KIE Model...</p>
                          </div>
                        ) : result ? (
                          <pre className="text-xs text-slate-700 font-mono font-medium whitespace-pre-wrap">
                            {JSON.stringify(result.extraction || result, null, 2)}
                          </pre>
                        ) : (
                          <div className="h-full flex items-center justify-center text-sm text-slate-400 font-semibold">
                            Waiting for prediction...
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ================= BATCH MODE UI ================= */}
        {appMode === "batch" && (
          <div className="flex flex-col gap-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
            
            <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xl shadow-slate-200/50">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-lg font-bold text-slate-800 flex items-center gap-2">
                  <Table className="w-5 h-5 text-blue-500" /> Batch Processing Queue
                </h2>
                
                <div className="flex gap-3">
                  <button onClick={clearBatchFiles} disabled={isBatchProcessing || batchFiles.length === 0}
                    className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-slate-200 text-red-600 hover:bg-red-50 text-sm font-semibold transition-colors disabled:opacity-50">
                    <Trash2 className="w-4 h-4" /> Clear Queue
                  </button>
                  <button onClick={() => batchInputRef.current?.click()} disabled={isBatchProcessing}
                    className="flex items-center gap-2 px-4 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 border border-slate-200 text-slate-700 text-sm font-bold transition-colors disabled:opacity-50">
                    <UploadCloud className="w-4 h-4" /> Add Files
                  </button>
                  <input type="file" ref={batchInputRef} onChange={handleBatchFileChange} className="hidden" accept="image/*" multiple />
                  
                  <button onClick={runBatchProcessing} disabled={isBatchProcessing || batchFiles.length === 0}
                    className="flex items-center gap-2 px-6 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-sm font-bold transition-colors shadow-sm disabled:opacity-50">
                    {isBatchProcessing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                    {isBatchProcessing ? "Processing..." : "Run Batch"}
                  </button>
                </div>
              </div>

              {batchFiles.length === 0 ? (
                <div 
                  onDragOver={handleDragOver}
                  onDrop={handleBatchDrop}
                  onClick={() => batchInputRef.current?.click()}
                  className="border-2 border-dashed border-slate-300 bg-slate-50 rounded-xl p-12 flex flex-col items-center justify-center cursor-pointer hover:bg-slate-100 transition-all"
                >
                  <div className="w-16 h-16 bg-white rounded-full flex items-center justify-center mb-4 shadow-sm border border-slate-200">
                    <UploadCloud className="w-8 h-8 text-slate-400" />
                  </div>
                  <p className="text-base font-bold text-slate-700">Drag & Drop Multiple Invoices Here</p>
                  <p className="text-sm text-slate-500 mt-1">Select multiple JPG, PNG, or PDF files to process in bulk</p>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="flex items-center justify-between text-sm text-slate-600 font-semibold">
                    <span>{batchFiles.length} files in queue</span>
                    {isBatchProcessing && <span className="text-blue-600">{batchProgress}% Completed</span>}
                  </div>
                  
                  {isBatchProcessing && (
                    <div className="w-full bg-slate-100 rounded-full h-2.5 overflow-hidden">
                      <div className="bg-blue-600 h-2.5 rounded-full transition-all duration-300" style={{ width: `${batchProgress}%` }}></div>
                    </div>
                  )}

                  <div className="border border-slate-200 rounded-xl overflow-hidden bg-white shadow-sm">
                    <div className="overflow-x-auto max-h-[500px]">
                      <table className="w-full text-sm text-left whitespace-nowrap">
                        <thead className="text-xs text-slate-500 uppercase bg-slate-50 sticky top-0 z-10 shadow-sm">
                          <tr>
                            <th className="px-6 py-3 font-bold">Status</th>
                            <th className="px-6 py-3 font-bold">File Name</th>
                            <th className="px-6 py-3 font-bold">Seller</th>
                            <th className="px-6 py-3 font-bold">Total Cost</th>
                            <th className="px-6 py-3 font-bold">Timestamp</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                          {batchFiles.map((file, index) => {
                            const res = batchResults[index];
                            return (
                              <tr key={index} className="hover:bg-slate-50">
                                <td className="px-6 py-4">
                                  {res ? (
                                    res.status === "Success" ? <span className="flex items-center gap-1.5 text-green-600 font-semibold"><CheckCircle2 className="w-4 h-4"/> Success</span>
                                    : <span className="flex items-center gap-1.5 text-red-600 font-semibold">Error</span>
                                  ) : isBatchProcessing && index === batchResults.length ? (
                                    <span className="flex items-center gap-1.5 text-blue-600 font-semibold"><Loader2 className="w-4 h-4 animate-spin"/> Processing</span>
                                  ) : (
                                    <span className="text-slate-400 font-medium">Pending</span>
                                  )}
                                </td>
                                <td className="px-6 py-4 font-medium text-slate-900">{file.name}</td>
                                <td className="px-6 py-4 text-slate-600 max-w-[200px] truncate">{res?.seller || "-"}</td>
                                <td className="px-6 py-4 text-slate-600 font-semibold text-green-700">{res?.totalCost || "-"}</td>
                                <td className="px-6 py-4 text-slate-600">{res?.timestamp || "-"}</td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {batchResults.length === batchFiles.length && batchFiles.length > 0 && (
                    <div className="flex justify-end pt-2">
                      <button onClick={exportCSV}
                        className="flex items-center gap-2 px-6 py-2.5 rounded-xl bg-green-600 hover:bg-green-700 text-white text-sm font-bold transition-colors shadow-md shadow-green-600/20">
                        <Download className="w-4 h-4" /> Export Results to CSV
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
            
          </div>
        )}

      </main>
    </div>
  );
}
