"use client";

import React, { useState, useRef } from "react";
import { Navbar } from "@/components/layout/Navbar";
import { useConfig } from "@/context/ConfigContext";
import { UploadCloud, Image as ImageIcon, Send, FileJson, Loader2, Bot, Layers, CheckCircle2, Download, Trash2, Play, XCircle, RotateCw } from "lucide-react";
import axios from "axios";

// Unified state for a single document
interface DocumentState {
  id: string;
  file: File;
  preview: string;
  status: "pending" | "processing" | "success" | "error";
  result: any | null;
  chatHistory: { role: "user" | "bot", content: string }[];
}

export default function Home() {
  const { apiUrl } = useConfig();
  
  // App State
  const [documents, setDocuments] = useState<DocumentState[]>([]);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  
  // Settings State
  const [ocrEngine, setOcrEngine] = useState("craft");
  const [kieModel, setKieModel] = useState("layoutlmv3");
  const [preprocess, setPreprocess] = useState(false);
  const [isProcessingQueue, setIsProcessingQueue] = useState(false);
  
  // Chat State (for the currently selected document)
  const [question, setQuestion] = useState("");
  const [isChatting, setIsChatting] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // ================= HANDLERS =================
  const handleFilesAdded = (files: File[]) => {
    const newDocs: DocumentState[] = files.map(f => ({
      id: Math.random().toString(36).substring(7),
      file: f,
      preview: URL.createObjectURL(f),
      status: "pending",
      result: null,
      chatHistory: []
    }));
    
    setDocuments(prev => {
      const updated = [...prev, ...newDocs];
      if (selectedIndex === null && updated.length > 0) {
        setTimeout(() => setSelectedIndex(prev.length), 0);
      }
      return updated;
    });
  };

  const onFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFilesAdded(Array.from(e.target.files));
    }
  };

  const handleDragOver = (e: React.DragEvent) => e.preventDefault();
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFilesAdded(Array.from(e.dataTransfer.files));
    }
  };

  const removeDocument = (index: number, e: React.MouseEvent) => {
    e.stopPropagation();
    setDocuments(prev => {
      const next = prev.filter((_, i) => i !== index);
      if (selectedIndex === index) {
        setSelectedIndex(null);
      } else if (selectedIndex !== null && selectedIndex > index) {
        setSelectedIndex(selectedIndex - 1);
      }
      return next;
    });
  };

  const rotateDocument = (index: number, e: React.MouseEvent) => {
    e.stopPropagation();
    const doc = documents[index];
    if (doc.status === "processing") return;
    
    const img = new window.Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.height;
      canvas.height = img.width;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.translate(canvas.width / 2, canvas.height / 2);
      ctx.rotate(90 * Math.PI / 180);
      ctx.drawImage(img, -img.width / 2, -img.height / 2);
      
      canvas.toBlob((blob) => {
        if (!blob) return;
        const newFile = new File([blob], doc.file.name, { type: doc.file.type || "image/jpeg" });
        setDocuments(prev => {
          const next = [...prev];
          next[index] = { 
            ...next[index], 
            file: newFile, 
            preview: URL.createObjectURL(newFile),
            status: "pending",
            result: null,
            chatHistory: []
          };
          return next;
        });
      }, doc.file.type || "image/jpeg");
    };
    img.src = doc.preview;
  };

  const clearAll = () => {
    setDocuments([]);
    setSelectedIndex(null);
  };

  // ================= PROCESSING LOGIC =================
  const processDocument = async (index: number) => {
    const doc = documents[index];
    if (!doc || doc.status === "processing") return;

    // Mark as processing
    setDocuments(prev => {
      const copy = [...prev];
      copy[index] = { ...copy[index], status: "processing", result: null };
      return copy;
    });

    let baseline = kieModel;
    if (kieModel === "phobert" && ocrEngine === "paddle") baseline = "phobert_paddle";
    if (kieModel === "layoutlmv3" && ocrEngine === "craft") baseline = "layoutlmv3_craft";

    const formData = new FormData();
    formData.append("file", doc.file);
    formData.append("baseline", baseline);
    formData.append("preprocess", preprocess ? "true" : "false");

    try {
      const res = await axios.post(`${apiUrl}/api/predict`, formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });
      if (res.data.success || res.data.extraction) {
        setDocuments(prev => {
          const copy = [...prev];
          copy[index] = { ...copy[index], status: "success", result: res.data };
          return copy;
        });
      } else {
        throw new Error("Prediction failed");
      }
    } catch (err) {
      setDocuments(prev => {
        const copy = [...prev];
        copy[index] = { ...copy[index], status: "error" };
        return copy;
      });
    }
  };

  const runQueue = async () => {
    setIsProcessingQueue(true);
    for (let i = 0; i < documents.length; i++) {
      if (documents[i].status === "pending" || documents[i].status === "error") {
        await processDocument(i);
      }
    }
    setIsProcessingQueue(false);
  };

  // ================= CHAT LOGIC =================
  const handleChat = async (e: React.FormEvent) => {
    e.preventDefault();
    if (selectedIndex === null || !question.trim() || isChatting) return;
    
    const doc = documents[selectedIndex];
    if (!doc || doc.status !== "success") return;

    const q = question;
    setQuestion("");
    setIsChatting(true);
    
    setDocuments(prev => {
      const copy = [...prev];
      copy[selectedIndex] = { 
        ...copy[selectedIndex], 
        chatHistory: [...copy[selectedIndex].chatHistory, { role: "user", content: q }]
      };
      return copy;
    });

    let baseline = kieModel;
    if (kieModel === "phobert" && ocrEngine === "paddle") baseline = "phobert_paddle";
    if (kieModel === "layoutlmv3" && ocrEngine === "craft") baseline = "layoutlmv3_craft";

    const formData = new FormData();
    formData.append("file", doc.file);
    formData.append("model", baseline);
    formData.append("question", q);
    formData.append("history", JSON.stringify(doc.chatHistory));

    try {
      const res = await axios.post(`${apiUrl}/api/chat`, formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });
      
      const answer = (res.data.success || res.data.answer) ? res.data.answer : "Sorry, an error occurred.";
      
      setDocuments(prev => {
        const copy = [...prev];
        copy[selectedIndex] = { 
          ...copy[selectedIndex], 
          chatHistory: [...copy[selectedIndex].chatHistory, { role: "bot", content: answer }]
        };
        return copy;
      });
    } catch (err) {
      setDocuments(prev => {
        const copy = [...prev];
        copy[selectedIndex] = { 
          ...copy[selectedIndex], 
          chatHistory: [...copy[selectedIndex].chatHistory, { role: "bot", content: "Error connecting to backend API." }]
        };
        return copy;
      });
    } finally {
      setIsChatting(false);
    }
  };

  // ================= EXPORT =================
  const exportCSV = () => {
    const successDocs = documents.filter(d => d.status === "success" && d.result);
    if (successDocs.length === 0) return;
    
    const headers = ["File Name", "Seller", "Address", "Timestamp", "Total Cost", "Items", "Status"];
    const csvRows = [headers.join(",")];

    for (const doc of successDocs) {
      const data = doc.result.extraction || doc.result;
      const values = [
        `"${doc.file.name.replace(/"/g, '""')}"`,
        `"${String(data.SELLER || data.seller || "").replace(/"/g, '""')}"`,
        `"${String(data.ADDRESS || data.address || "").replace(/"/g, '""')}"`,
        `"${String(data.TIMESTAMP || data.timestamp || "").replace(/"/g, '""')}"`,
        `"${String(data.TOTAL_COST || data.total_cost || "").replace(/"/g, '""')}"`,
        `"${String(JSON.stringify(data.ITEMS || data.items || [])).replace(/"/g, '""')}"`,
        `"${doc.status}"`
      ];
      csvRows.push(values.join(","));
    }
    
    const blob = new Blob(["\uFEFF" + csvRows.join("\n")], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", `avir_kie_export_${new Date().getTime()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const selectedDoc = selectedIndex !== null && documents[selectedIndex] ? documents[selectedIndex] : null;

  return (
    <div className="flex-1 flex flex-col h-screen bg-slate-50 overflow-hidden">
      <Navbar />
      
      {/* TOP CONTROLS */}
      <div className="bg-white border-b border-slate-200 px-6 py-3 flex items-center justify-between shrink-0 shadow-sm z-10">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <label className="text-xs font-bold text-slate-500 uppercase tracking-wider">OCR:</label>
            <select 
              value={ocrEngine} 
              onChange={e => {
                setOcrEngine(e.target.value);
                setDocuments(prev => prev.map(d => ({ ...d, status: "pending", result: null, chatHistory: [] })));
              }}
              disabled={kieModel === "qwen2_vl" || kieModel === "minicpm_v"}
              className="bg-slate-50 border border-slate-200 rounded-lg px-3 py-1.5 text-sm text-slate-900 font-medium focus:outline-none focus:border-blue-500 disabled:opacity-50"
            >
              <option value="craft">CRAFT + VietOCR</option>
              <option value="paddle">PaddleOCR</option>
              <option value="none" disabled hidden>Built-in VLM</option>
            </select>
          </div>

          <div className="flex items-center gap-2">
            <label className="text-xs font-bold text-slate-500 uppercase tracking-wider">Model:</label>
            <select 
              value={kieModel} 
              onChange={e => {
                const newModel = e.target.value;
                setKieModel(newModel);
                if (newModel === "qwen2_vl" || newModel === "minicpm_v") setOcrEngine("none");
                else if (ocrEngine === "none") setOcrEngine("craft");
                setDocuments(prev => prev.map(d => ({ ...d, status: "pending", result: null, chatHistory: [] })));
              }}
              className="bg-slate-50 border border-slate-200 rounded-lg px-3 py-1.5 text-sm text-slate-900 font-medium focus:outline-none focus:border-blue-500"
            >
              <option value="rule_based">Rule-based (Heuristic)</option>
              <option value="phobert">PhoBERT</option>
              <option value="layoutlmv3">LayoutLMv3</option>
              <option value="qwen2_vl">Qwen2-VL</option>
              <option value="minicpm_v">MiniCPM-V</option>
            </select>
          </div>
          <label className="flex items-center gap-2 cursor-pointer border-l border-slate-200 pl-4">
            <input 
              type="checkbox" 
              checked={preprocess} 
              onChange={e => setPreprocess(e.target.checked)}
              className="rounded border-slate-300 text-blue-600 focus:ring-blue-500 w-4 h-4" 
            />
            <span className="text-sm font-bold text-slate-600">Auto Enhance</span>
          </label>
        </div>
        
        <div className="flex items-center gap-3">
          <button 
            onClick={() => fileInputRef.current?.click()} 
            disabled={isProcessingQueue}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-100 hover:bg-slate-200 border border-slate-200 text-slate-700 text-sm font-bold transition-colors disabled:opacity-50"
          >
            <UploadCloud className="w-4 h-4" /> Add Files
          </button>
          <input type="file" ref={fileInputRef} onChange={onFileInputChange} className="hidden" accept="image/*" multiple />
          
          <button 
            onClick={runQueue} 
            disabled={isProcessingQueue || documents.length === 0}
            className="flex items-center gap-2 px-6 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white text-sm font-bold transition-colors shadow-sm disabled:opacity-50"
          >
            {isProcessingQueue ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            {isProcessingQueue ? "Processing..." : "Run All Pending"}
          </button>
          
          <button 
            onClick={exportCSV} 
            disabled={documents.filter(d => d.status === 'success').length === 0}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-green-600 hover:bg-green-700 text-white text-sm font-bold transition-colors shadow-sm disabled:opacity-50"
          >
            <Download className="w-4 h-4" /> Export
          </button>
        </div>
      </div>

      <main className="flex-1 flex overflow-hidden">
        
        {/* LEFT SIDEBAR - QUEUE */}
        <div className="w-80 bg-white border-r border-slate-200 flex flex-col shrink-0 z-0 shadow-sm">
          <div className="p-4 border-b border-slate-100 flex items-center justify-between bg-slate-50/50">
            <h2 className="text-sm font-bold text-slate-800 flex items-center gap-2">
              <Layers className="w-4 h-4 text-blue-500" /> Queue
            </h2>
            <span className="text-xs font-semibold bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full">
              {documents.length} files
            </span>
          </div>

          <div 
            className="flex-1 overflow-y-auto p-3 space-y-2"
            onDragOver={handleDragOver}
            onDrop={handleDrop}
          >
            {documents.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-slate-400 p-6 text-center border-2 border-dashed border-slate-200 rounded-xl bg-slate-50">
                <UploadCloud className="w-10 h-10 mb-3 opacity-30" />
                <p className="text-sm font-medium text-slate-500">Drag & drop invoices here</p>
              </div>
            ) : (
              documents.map((doc, idx) => (
                <div 
                  key={doc.id}
                  onClick={() => setSelectedIndex(idx)}
                  className={`relative group flex items-center gap-3 p-2.5 rounded-xl border cursor-pointer transition-all ${
                    selectedIndex === idx 
                      ? "bg-blue-50 border-blue-300 shadow-sm" 
                      : "bg-white border-slate-200 hover:border-blue-200 hover:bg-slate-50"
                  }`}
                >
                  <img src={doc.preview} className="w-10 h-10 rounded object-cover border border-slate-200 bg-white shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-semibold text-slate-800 truncate">{doc.file.name}</p>
                    <div className="flex items-center gap-1.5 mt-0.5">
                      {doc.status === "pending" && <span className="text-[10px] uppercase font-bold text-slate-500">Pending</span>}
                      {doc.status === "processing" && <span className="text-[10px] uppercase font-bold text-blue-600 flex items-center gap-1"><Loader2 className="w-3 h-3 animate-spin"/> Processing</span>}
                      {doc.status === "success" && <span className="text-[10px] uppercase font-bold text-green-600 flex items-center gap-1"><CheckCircle2 className="w-3 h-3"/> Success</span>}
                      {doc.status === "error" && <span className="text-[10px] uppercase font-bold text-red-600 flex items-center gap-1"><XCircle className="w-3 h-3"/> Error</span>}
                    </div>
                  </div>
                  <div className="opacity-0 group-hover:opacity-100 flex items-center gap-1 transition-all">
                    <button 
                      onClick={(e) => rotateDocument(idx, e)}
                      title="Rotate 90°"
                      className="p-1.5 text-slate-400 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-all"
                    >
                      <RotateCw className="w-4 h-4" />
                    </button>
                    <button 
                      onClick={(e) => removeDocument(idx, e)}
                      title="Remove"
                      className="p-1.5 text-slate-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-all"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
          
          {documents.length > 0 && (
            <div className="p-3 border-t border-slate-200 bg-white">
              <button 
                onClick={clearAll}
                disabled={isProcessingQueue}
                className="w-full py-2 text-sm font-semibold text-red-600 hover:bg-red-50 rounded-lg border border-transparent transition-colors disabled:opacity-50"
              >
                Clear Queue
              </button>
            </div>
          )}
        </div>

        {/* RIGHT MAIN AREA - DETAIL VIEW */}
        <div className="flex-1 bg-slate-50/50 flex flex-col p-6 overflow-hidden">
          {!selectedDoc ? (
            <div className="h-full flex flex-col items-center justify-center text-slate-400 bg-white border border-slate-200 rounded-2xl shadow-sm">
              <ImageIcon className="w-16 h-16 mb-4 opacity-20" />
              <p className="text-base font-semibold text-slate-600">No document selected</p>
              <p className="text-sm mt-1">Upload files and select one from the queue to view details.</p>
            </div>
          ) : (
            <div className="flex gap-6 h-full min-h-0">
              
              {/* Image Preview & VQA */}
              <div className="flex-1 flex flex-col gap-6 min-w-0">
                {/* PREVIEW */}
                <div className="flex-1 bg-white border border-slate-200 rounded-2xl shadow-sm overflow-hidden flex flex-col relative">
                  <div className="bg-slate-50 px-4 py-2.5 border-b border-slate-200 text-xs font-bold text-slate-500 uppercase tracking-wider flex justify-between items-center">
                    <span className="truncate pr-4">Document View: {selectedDoc.file.name}</span>
                    <button onClick={() => processDocument(selectedIndex!)} disabled={selectedDoc.status === 'processing'} className="shrink-0 text-blue-600 hover:text-blue-700 bg-blue-50 px-3 py-1 rounded-full border border-blue-100 flex items-center gap-1 transition-colors disabled:opacity-50">
                      {selectedDoc.status === 'processing' ? <Loader2 className="w-3 h-3 animate-spin" /> : <Play className="w-3 h-3" />}
                      {selectedDoc.status === 'processing' ? 'Running' : selectedDoc.status === 'success' ? 'Rerun' : 'Run Analyze'}
                    </button>
                  </div>
                  <div className="flex-1 p-4 flex items-center justify-center bg-slate-100/50 overflow-auto">
                    <img 
                      src={selectedDoc.result?.image_url ? (selectedDoc.result.image_url.startsWith('http') ? selectedDoc.result.image_url : `${apiUrl}${selectedDoc.result.image_url}`) : selectedDoc.preview} 
                      alt="Document" 
                      className="max-w-full max-h-full object-contain rounded border border-slate-200 bg-white shadow-sm"
                    />
                  </div>
                </div>

                {/* VQA CHAT */}
                <div className={`bg-white border border-slate-200 rounded-2xl p-4 shadow-sm h-64 flex flex-col shrink-0 transition-all ${
                  (kieModel === "qwen2_vl" || kieModel === "minicpm_v") ? "" : "opacity-60 pointer-events-none grayscale-[50%]"
                }`}>
                  <div className="flex justify-between items-center mb-3">
                    <h2 className="text-sm font-bold text-slate-800 flex items-center gap-1.5">
                      <Bot className="w-4 h-4 text-indigo-500" /> Visual Q&A
                    </h2>
                    <div className="text-[10px] uppercase font-bold text-slate-500 bg-slate-100 px-2 py-0.5 rounded">
                       {(kieModel === "qwen2_vl" || kieModel === "minicpm_v") ? (kieModel === "qwen2_vl" ? "Qwen2-VL" : "MiniCPM-V") : "Locked"}
                    </div>
                  </div>

                  <div className="flex-1 overflow-y-auto mb-3 space-y-3 pr-2">
                    {selectedDoc.chatHistory.length === 0 ? (
                      <div className="h-full flex flex-col items-center justify-center text-slate-400">
                        {!(kieModel === "qwen2_vl" || kieModel === "minicpm_v") ? (
                          <p className="text-xs font-semibold text-red-400 text-center">Select Qwen2-VL or MiniCPM-V<br/>to enable Chat.</p>
                        ) : (
                          <p className="text-xs font-medium text-center">Ask a question about this specific document.</p>
                        )}
                      </div>
                    ) : (
                      selectedDoc.chatHistory.map((msg, i) => (
                        <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                          <div className={`max-w-[90%] rounded-xl px-3 py-2 text-xs font-medium shadow-sm ${
                            msg.role === "user" ? "bg-indigo-600 text-white rounded-br-none" : "bg-slate-100 text-slate-800 border border-slate-200 rounded-bl-none"
                          }`}>
                            {msg.content}
                          </div>
                        </div>
                      ))
                    )}
                    {isChatting && (
                      <div className="flex justify-start">
                        <div className="bg-slate-100 border border-slate-200 rounded-xl rounded-bl-none px-3 py-2 flex items-center gap-1 shadow-sm">
                          <Loader2 className="w-3 h-3 text-indigo-500 animate-spin" />
                        </div>
                      </div>
                    )}
                  </div>

                  <form onSubmit={handleChat} className="relative mt-auto">
                    <input 
                      type="text" 
                      value={question}
                      onChange={e => setQuestion(e.target.value)}
                      placeholder="Ask anything..."
                      disabled={selectedDoc.status !== 'success' || isChatting}
                      className="w-full bg-slate-50 border border-slate-200 rounded-lg pl-3 pr-10 py-2 text-slate-900 text-xs font-medium focus:outline-none focus:border-indigo-500 disabled:opacity-50"
                    />
                    <button 
                      type="submit"
                      disabled={selectedDoc.status !== 'success' || !question.trim() || isChatting}
                      className="absolute right-1 top-1 p-1 bg-indigo-600 text-white rounded shadow-sm hover:bg-indigo-700 transition-colors disabled:opacity-50"
                    >
                      <Send className="w-3 h-3" />
                    </button>
                  </form>
                </div>
              </div>

              {/* JSON Extraction */}
              <div className="w-[400px] xl:w-[450px] bg-white border border-slate-200 rounded-2xl shadow-sm flex flex-col shrink-0">
                <div className="bg-slate-50 px-4 py-2.5 border-b border-slate-200 text-xs font-bold text-slate-500 uppercase tracking-wider flex items-center gap-2">
                  <FileJson className="w-4 h-4 text-purple-600" /> Extracted Data
                </div>
                <div className="flex-1 p-4 overflow-auto">
                  {selectedDoc.status === "processing" ? (
                    <div className="h-full flex flex-col items-center justify-center space-y-3">
                      <Loader2 className="w-8 h-8 text-blue-600 animate-spin" />
                      <p className="text-sm font-semibold text-slate-500 animate-pulse">Running KIE Model...</p>
                    </div>
                  ) : selectedDoc.result ? (
                    <div className="space-y-4">
                      {(() => {
                        const data = selectedDoc.result.extraction || selectedDoc.result;
                        return (
                          <div className="text-xs text-slate-700">
                            <div className="grid grid-cols-3 gap-2 bg-slate-50 p-3 rounded-lg border border-slate-200 mb-3 shadow-sm">
                              <div className="font-semibold text-slate-500">SELLER:</div>
                              <div className="col-span-2 font-medium text-slate-800">{data.SELLER || data.seller || "N/A"}</div>
                              
                              <div className="font-semibold text-slate-500">ADDRESS:</div>
                              <div className="col-span-2 font-medium text-slate-800">{data.ADDRESS || data.address || "N/A"}</div>
                              
                              <div className="font-semibold text-slate-500">TIMESTAMP:</div>
                              <div className="col-span-2 font-medium text-slate-800">{data.TIMESTAMP || data.timestamp || "N/A"}</div>
                              
                              <div className="font-semibold text-slate-500">TOTAL COST:</div>
                              <div className="col-span-2 font-bold text-indigo-600">{data.TOTAL_COST || data.total_cost || "N/A"}</div>
                            </div>
                            
                            <div className="font-bold text-slate-600 mb-2 uppercase tracking-wider flex items-center gap-1.5">
                              <Layers className="w-3.5 h-3.5 text-blue-500" />
                              Items List
                            </div>
                            <div className="border border-slate-200 rounded-lg overflow-hidden shadow-sm">
                              <table className="w-full text-left border-collapse bg-white">
                                <thead>
                                  <tr className="bg-slate-100 text-slate-500 border-b border-slate-200">
                                    <th className="px-2 py-2.5 font-semibold w-12 text-center">Qty</th>
                                    <th className="px-2 py-2.5 font-semibold">Name</th>
                                    <th className="px-2 py-2.5 font-semibold text-right">Price</th>
                                    <th className="px-2 py-2.5 font-semibold text-right">Amount</th>
                                  </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-100">
                                  {(data.ITEMS || data.items || []).length > 0 ? (
                                    (data.ITEMS || data.items || []).map((item: any, idx: number) => (
                                      <tr key={idx} className="hover:bg-slate-50 transition-colors">
                                        <td className="px-2 py-2 align-middle text-center font-medium bg-slate-50/50">{item.ITEM_QTY || item.item_qty || item.SL || "-"}</td>
                                        <td className="px-2 py-2 align-middle font-medium text-slate-800">{item.ITEM_NAME || item.item_name || "-"}</td>
                                        <td className="px-2 py-2 align-middle text-right text-slate-500">{item.ITEM_PRICE || item.item_price || "-"}</td>
                                        <td className="px-2 py-2 align-middle text-right font-semibold text-slate-700 bg-slate-50/50">{item.ITEM_AMOUNT || item.item_amount || "-"}</td>
                                      </tr>
                                    ))
                                  ) : (
                                    <tr>
                                      <td colSpan={4} className="p-6 text-center text-slate-400 italic font-medium">
                                        No items extracted
                                      </td>
                                    </tr>
                                  )}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        );
                      })()}
                    </div>
                  ) : selectedDoc.status === "error" ? (
                     <div className="h-full flex items-center justify-center text-sm text-red-500 font-semibold text-center p-4">
                        Analysis Failed.<br/>Please check the backend logs.
                     </div>
                  ) : (
                    <div className="h-full flex flex-col items-center justify-center text-sm text-slate-400 font-medium text-center p-4">
                      Click "Run Analyze" on the image<br/>or "Run All Pending" to extract data.
                    </div>
                  )}
                </div>
              </div>

            </div>
          )}
        </div>
      </main>
    </div>
  );
}
