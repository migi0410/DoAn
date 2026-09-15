"use client";

import React, { useState, useRef, useCallback, useEffect } from "react";
import axios from "axios";
import {
  UploadCloud, FileImage, Settings2, Play, FileJson,
  Loader2, CheckCircle2, MessageSquare, GitCompare,
  ChevronRight, Send, Bot, User, X,
  RefreshCw, Download, Edit3, Check, AlertTriangle, Coffee, ShoppingBag,
  Store, FileText, CheckCircle, Camera, ZoomIn, ZoomOut, RotateCw,
  Maximize2, Plus, Trash2, Copy, Save, History, Archive, Search, Eye, CheckCheck
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

// Proxied by Next.js rewrites directly to FastAPI backend on port 8000
const API_BASE = "";

// --- CLEAN MODEL DEFINITIONS ---
const MODELS = [
  {
    value: "qwen3_lora_v2",
    name: "Qwen3-VL 8B (LoRA v2) - Mô hình chính",
    isProposed: true,
    badge: "Mô hình đề xuất",
    desc: "Mô hình Vision-Language được huấn luyện tối ưu trích xuất tiếng Việt đa dòng và phân cấp."
  },
  {
    value: "qwen3_base_v2",
    name: "Qwen3-VL 8B (Base Foundation)",
    isProposed: false,
    badge: "Gốc",
    desc: "Mô hình thị giác ngôn ngữ gốc chưa qua tinh chỉnh (Zero-shot)."
  },
  {
    value: "deepseek_qwen25",
    name: "DeepSeek-OCR + Qwen2.5 (7B)",
    isProposed: false,
    badge: "OCR + LLM",
    desc: "Pipeline hai giai đoạn kết hợp nhận diện văn bản OCR và mô hình ngôn ngữ lớn."
  },
  {
    value: "deepseek_regex",
    name: "DeepSeek-OCR + Heuristic Regex",
    isProposed: false,
    badge: "Truyền thống",
    desc: "Phương pháp hai giai đoạn truyền thống sử dụng OCR và biểu thức chính quy (Regex)."
  },
  {
    value: "minicpm_v",
    name: "MiniCPM-V 2.6 (8B)",
    isProposed: false,
    badge: "Đối sánh",
    desc: "Mô hình VLM mã nguồn mở dùng để đối sánh."
  }
];

const PRESET_SAMPLES = [
  { id: "sample_winmart", name: "Siêu Thị WinMart", icon: ShoppingBag, desc: "Hóa đơn nhiều món, rớt dòng", file: "winmart_template.jpg" },
  { id: "sample_highland", name: "Highlands Coffee", icon: Coffee, desc: "F&B, khuyết đơn giá", file: "highland_template.jpg" },
  { id: "sample_circlek", name: "Circle K", icon: Store, desc: "Giấy in nhiệt POS", file: "circle_k_template.webp" },
  { id: "sample_phuclong", name: "Phúc Long", icon: Coffee, desc: "Trà & Cà phê", file: "phuc_long_template.jpg" },
  { id: "sample_viettel", name: "Viettel e-Invoice", icon: FileText, desc: "Hóa đơn điện tử A4 có VAT", file: "viettel_template.jpg" },
];

const SUGGESTED_QUESTIONS = [
  "Tổng thanh toán hóa đơn là bao nhiêu?",
  "Có bao nhiêu mặt hàng trong hóa đơn?",
  "Món hàng nào có giá trị cao nhất?",
  "Địa chỉ của cửa hàng ở đâu?",
  "Hóa đơn có ghi nhận thuế VAT không?",
];

type TabType = "extract" | "chat" | "compare" | "history";
type Message = { role: "user" | "assistant"; content: string };

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [selectedSampleId, setSelectedSampleId] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [model, setModel] = useState("qwen3_lora_v2");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<Record<string, any> | null>(null);
  const [validation, setValidation] = useState<any>(null);
  const [latency, setLatency] = useState<number | null>(null);
  const [activeTab, setActiveTab] = useState<TabType>("extract");
  const [isServerOnline, setIsServerOnline] = useState<boolean | null>(null);

  // Edit table state
  const [isEditing, setIsEditing] = useState(false);
  const [editableData, setEditableData] = useState<Record<string, any>>({});
  const [copySuccess, setCopySuccess] = useState(false);
  const [saveSuccessMsg, setSaveSuccessMsg] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);

  // History ledger state
  const [historyRecords, setHistoryRecords] = useState<any[]>([]);
  const [historySummary, setHistorySummary] = useState<any>(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historySearch, setHistorySearch] = useState("");

  // Image viewer state
  const [zoomLevel, setZoomLevel] = useState(1);
  const [rotation, setRotation] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Chat state
  const [messages, setMessages] = useState<Message[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Compare state
  const [compareModel, setCompareModel] = useState("deepseek_regex");
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareResult, setCompareResult] = useState<{ left: any; right: any } | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);

  // Check server health
  useEffect(() => {
    const checkServer = async () => {
      try {
        const res = await axios.get(`${API_BASE}/`, { timeout: 3000 });
        setIsServerOnline(res.status === 200);
      } catch {
        setIsServerOnline(false);
      }
    };
    checkServer();
    const interval = setInterval(checkServer, 8000);
    return () => clearInterval(interval);
  }, []);

  // Update editable data when new result arrives
  useEffect(() => {
    if (result) {
      setEditableData(JSON.parse(JSON.stringify(result)));
      setIsEditing(false);
    }
  }, [result]);

  const executePredict = async (
    targetSampleId?: string | null,
    targetFile?: File | null,
    targetModel?: string
  ) => {
    const sid = targetSampleId !== undefined ? targetSampleId : selectedSampleId;
    const f = targetFile !== undefined ? targetFile : file;
    const m = targetModel || model;

    if (!f && !sid) return;
    setLoading(true);
    setResult(null);
    setValidation(null);

    const fd = new FormData();
    if (f) {
      fd.append("file", f);
    } else if (sid) {
      fd.append("sample_id", sid);
    }
    fd.append("model", m);
    fd.append("schema_version", "v2");

    try {
      const res = await axios.post(`${API_BASE}/api/predict`, fd);
      setResult(res.data.extraction);
      setValidation(res.data.validation);
      setLatency(res.data.latency_seconds);
    } catch (e: any) {
      alert("Lỗi trích xuất: " + (e.response?.data?.detail || e.message));
    } finally {
      setLoading(false);
    }
  };

  const handlePredict = () => {
    executePredict();
  };

  const handleFile = useCallback((f: File) => {
    setFile(f);
    setSelectedSampleId(null);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setValidation(null);
    setLatency(null);
    setMessages([]);
    setCompareResult(null);
    setZoomLevel(1);
    setRotation(0);
    executePredict(null, f);
  }, [model]);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);
  };

  const handleSelectSample = (sample: typeof PRESET_SAMPLES[0]) => {
    setSelectedSampleId(sample.id);
    setFile(null);
    const imgUrl = `${API_BASE}/templates_images/${sample.file}`;
    setPreview(imgUrl);
    setResult(null);
    setValidation(null);
    setLatency(null);
    setMessages([]);
    setCompareResult(null);
    setZoomLevel(1);
    setRotation(0);
    executePredict(sample.id, null);
  };

  // Auto-load first sample (WinMart) on page mount so the demo is immediately active
  useEffect(() => {
    const defaultSample = PRESET_SAMPLES[0];
    setSelectedSampleId(defaultSample.id);
    setPreview(`${API_BASE}/templates_images/${defaultSample.file}`);
    executePredict(defaultSample.id, null);
  }, []);

  const handleChat = async () => {
    if (!chatInput.trim() || (!file && !selectedSampleId)) return;
    const userMsg: Message = { role: "user", content: chatInput };
    setMessages((m) => [...m, userMsg]);
    const q = chatInput;
    setChatInput("");
    setChatLoading(true);

    const fd = new FormData();
    fd.append("question", q);
    fd.append("model", model);
    if (result) {
      fd.append("context_json", JSON.stringify(editableData || result));
    }

    try {
      const res = await axios.post(`${API_BASE}/api/chat`, fd);
      setMessages((m) => [...m, { role: "assistant", content: res.data.answer }]);
    } catch (e: any) {
      setMessages((m) => [...m, { role: "assistant", content: "⚠️ Không thể kết nối với server. Vui lòng thử lại." }]);
    } finally {
      setChatLoading(false);
      setTimeout(() => chatEndRef.current?.scrollIntoView({ behavior: "smooth" }), 100);
    }
  };

  const handleCompare = async () => {
    if (!file && !selectedSampleId) return;
    setCompareLoading(true);
    setCompareResult(null);

    const fd1 = new FormData();
    if (file) fd1.append("file", file);
    if (selectedSampleId) fd1.append("sample_id", selectedSampleId);
    fd1.append("model", model);
    fd1.append("schema_version", "v2");

    const fd2 = new FormData();
    if (file) fd2.append("file", file);
    if (selectedSampleId) fd2.append("sample_id", selectedSampleId);
    fd2.append("model", compareModel);
    fd2.append("schema_version", "v2");

    try {
      const [r1, r2] = await Promise.all([
        axios.post(`${API_BASE}/api/predict`, fd1),
        axios.post(`${API_BASE}/api/predict`, fd2),
      ]);
      setCompareResult({ left: r1.data, right: r2.data });
    } catch (e: any) {
      alert("Lỗi so sánh: " + e.message);
    } finally {
      setCompareLoading(false);
    }
  };

  // History ledger API handlers
  const fetchHistory = useCallback(async () => {
    try {
      setHistoryLoading(true);
      const res = await axios.get(`${API_BASE}/api/history`);
      if (res.data.success) {
        setHistoryRecords(res.data.records || []);
        setHistorySummary(res.data.summary || null);
      }
    } catch (e) {
      console.error("Error fetching history:", e);
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  const handleSaveReceipt = async () => {
    if (!editableData || !editableData.TOTAL_COST) return;
    setIsSaving(true);
    try {
      const payload = {
        seller: editableData.SELLER || "Hóa đơn bán lẻ",
        address: editableData.ADDRESS || "",
        timestamp: editableData.TIMESTAMP || "",
        total_cost: String(editableData.TOTAL_COST || "0"),
        items: items,
        is_valid: isMatch,
        discrepancy: Math.abs(itemsSum - declaredTotal),
        model_id: model,
        image_url: preview || "",
        notes: isMatch ? "Đã đối soát khớp 100%" : `Lệch ${Math.abs(itemsSum - declaredTotal).toLocaleString()} đ`
      };
      const res = await axios.post(`${API_BASE}/api/history/save`, payload);
      if (res.data.success) {
        setSaveSuccessMsg(`✓ Đã lưu thành công vào Sổ Kế Toán! Mã: #${res.data.id}`);
        setTimeout(() => setSaveSuccessMsg(null), 4000);
        fetchHistory();
      }
    } catch (e: any) {
      alert("Lỗi lưu hóa đơn: " + (e.response?.data?.detail || e.message));
    } finally {
      setIsSaving(false);
    }
  };

  const handleDeleteHistory = async (id: string) => {
    if (!confirm(`Bạn có chắc muốn xóa bản ghi #${id} khỏi Sổ Kế Toán?`)) return;
    try {
      await axios.delete(`${API_BASE}/api/history/${id}`);
      fetchHistory();
    } catch (e: any) {
      alert("Lỗi xóa bản ghi: " + e.message);
    }
  };

  const handleLoadFromHistory = (rec: any) => {
    setSelectedSampleId(null);
    setFile(null);
    if (rec.image_url) setPreview(rec.image_url);
    const loadedData = {
      SELLER: rec.seller,
      ADDRESS: rec.address,
      TIMESTAMP: rec.receipt_time,
      TOTAL_COST: rec.total_cost,
      ITEMS: rec.items
    };
    setResult(loadedData);
    setEditableData(loadedData);
    setActiveTab("extract");
  };

  // Extract items helper
  const getItemsList = () => {
    const rawItems = editableData.ITEMS;
    if (Array.isArray(rawItems)) {
      return rawItems.map((it: any) => ({
        name: it.name || it.ITEM_NAME || "",
        qty: String(it.qty || it.ITEM_QTY || "1"),
        price: String(it.price || it.ITEM_PRICE || ""),
        amount: String(it.amount || it.ITEM_AMOUNT || ""),
      }));
    }
    return [];
  };

  const items = getItemsList();

  // Handle manual edits
  const handleFieldChange = (key: string, val: string) => {
    setEditableData((prev) => ({ ...prev, [key]: val }));
  };

  const handleItemChange = (idx: number, field: string, val: string) => {
    const updated = [...items];
    updated[idx] = { ...updated[idx], [field]: val };
    setEditableData((prev) => ({ ...prev, ITEMS: updated }));
  };

  const handleAddItem = () => {
    const updated = [...items, { name: "Mặt hàng mới", qty: "1", price: "", amount: "" }];
    setEditableData((prev) => ({ ...prev, ITEMS: updated }));
  };

  const handleDeleteItem = (idx: number) => {
    const updated = items.filter((_, i) => i !== idx);
    setEditableData((prev) => ({ ...prev, ITEMS: updated }));
  };

  // Calculate live sum from items
  const calculateItemsTotal = () => {
    let sum = 0;
    for (const it of items) {
      const numStr = String(it.amount || "").replace(/[^0-9]/g, "");
      const num = parseFloat(numStr);
      if (!isNaN(num)) sum += num;
    }
    return sum;
  };

  const declaredTotalNum = () => {
    const numStr = String(editableData.TOTAL_COST || "").replace(/[^0-9]/g, "");
    const num = parseFloat(numStr);
    return isNaN(num) ? 0 : num;
  };

  const itemsSum = calculateItemsTotal();
  const declaredTotal = declaredTotalNum();
  const isMatch = itemsSum > 0 && declaredTotal > 0 && Math.abs(itemsSum - declaredTotal) < 2;

  // Export handlers
  const exportCSV = () => {
    let csv = "\uFEFF"; // UTF-8 BOM for Excel support in Vietnamese
    csv += "THÔNG TIN HÓA ĐƠN\n";
    csv += `Đơn vị bán hàng,"${editableData.SELLER || ""}"\n`;
    csv += `Địa chỉ,"${editableData.ADDRESS || ""}"\n`;
    csv += `Thời gian lập,"${editableData.TIMESTAMP || ""}"\n`;
    csv += `Tổng thanh toán,"${editableData.TOTAL_COST || ""}"\n\n`;
    csv += "STT,Tên mặt hàng,Số lượng,Đơn giá (VNĐ),Thành tiền (VNĐ)\n";
    items.forEach((it, idx) => {
      csv += `${idx + 1},"${(it.name || "").replace(/"/g, '""')}","${it.qty}","${it.price}","${it.amount}"\n`;
    });

    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `HoaDon_${Date.now()}.csv`;
    link.click();
  };

  const exportJSON = () => {
    const blob = new Blob([JSON.stringify(editableData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `HoaDon_${Date.now()}.json`;
    link.click();
  };

  const copyJSONToClipboard = () => {
    navigator.clipboard.writeText(JSON.stringify(editableData, null, 2));
    setCopySuccess(true);
    setTimeout(() => setCopySuccess(false), 2000);
  };

  const handlePrint = () => {
    window.print();
  };

  const selectedModelInfo = MODELS.find((m) => m.value === model) || MODELS[0];

  return (
    <div className="min-h-screen bg-[#0a0d14] text-slate-100 font-sans selection:bg-indigo-500/30 pb-16">
      {/* ── TOP NAVIGATION BAR ── */}
      <nav className="border-b border-white/10 bg-[#0d121d]/80 backdrop-blur-md sticky top-0 z-40 px-4 sm:px-8 py-3.5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center shadow-md shadow-indigo-500/20">
            <FileText className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="font-bold text-base tracking-tight text-white flex items-center gap-2">
              AVIR-KIE
              <span className="text-[11px] font-semibold px-2 py-0.5 rounded-md bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                Demo
              </span>
            </div>
            <div className="text-[11px] text-slate-400 hidden sm:block">
              Trích Xuất Dữ Liệu Hóa Đơn Tự Động bằng Vision-Language Model
            </div>
          </div>
        </div>

        {/* Simple Status Pill */}
        <div className="flex items-center gap-3">
          <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium border ${
            isServerOnline
              ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-300"
              : isServerOnline === false
              ? "bg-rose-500/10 border-rose-500/30 text-rose-300"
              : "bg-slate-500/10 border-slate-500/30 text-slate-400"
          }`}>
            <span className={`w-2 h-2 rounded-full ${
              isServerOnline ? "bg-emerald-400 animate-pulse shadow-[0_0_8px_#34d399]" : isServerOnline === false ? "bg-rose-400" : "bg-slate-400"
            }`} />
            <span>{isServerOnline ? "Hệ thống AI: Sẵn sàng" : isServerOnline === false ? "Hệ thống: Ngoại tuyến" : "Đang kiểm tra..."}</span>
          </div>
        </div>
      </nav>

      {/* ── MAIN CONTAINER ── */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 pt-6">
        {/* Navigation Tabs */}
        <div className="flex items-center gap-2 mb-6 border-b border-white/10 pb-3 overflow-x-auto">
          <button
            onClick={() => setActiveTab("extract")}
            className={`px-4 py-2 rounded-xl text-sm font-medium transition-all flex items-center gap-2 shrink-0 ${
              activeTab === "extract"
                ? "bg-indigo-600 text-white shadow-lg shadow-indigo-600/20"
                : "text-slate-400 hover:text-white hover:bg-white/5"
            }`}
          >
            <FileJson className="w-4 h-4" /> Trích Xuất Hóa Đơn
          </button>
          <button
            onClick={() => setActiveTab("compare")}
            className={`px-4 py-2 rounded-xl text-sm font-medium transition-all flex items-center gap-2 shrink-0 ${
              activeTab === "compare"
                ? "bg-indigo-600 text-white shadow-lg shadow-indigo-600/20"
                : "text-slate-400 hover:text-white hover:bg-white/5"
            }`}
          >
            <GitCompare className="w-4 h-4" /> So Sánh 2 Mô Hình
          </button>
          <button
            onClick={() => setActiveTab("chat")}
            className={`px-4 py-2 rounded-xl text-sm font-medium transition-all flex items-center gap-2 shrink-0 ${
              activeTab === "chat"
                ? "bg-indigo-600 text-white shadow-lg shadow-indigo-600/20"
                : "text-slate-400 hover:text-white hover:bg-white/5"
            }`}
          >
            <MessageSquare className="w-4 h-4" /> Hỏi Đáp (Q&A)
          </button>
          <button
            onClick={() => {
              setActiveTab("history");
              fetchHistory();
            }}
            className={`px-4 py-2 rounded-xl text-sm font-medium transition-all flex items-center gap-2 shrink-0 ${
              activeTab === "history"
                ? "bg-indigo-600 text-white shadow-lg shadow-indigo-600/20"
                : "text-slate-400 hover:text-white hover:bg-white/5"
            }`}
          >
            <Archive className="w-4 h-4" /> Sổ Lưu Trữ Hóa Đơn
            {historyRecords.length > 0 && (
              <span className="text-[10px] font-bold px-1.5 py-0.2 rounded-full bg-white/20 text-white ml-0.5">
                {historyRecords.length}
              </span>
            )}
          </button>
        </div>

        {/* Tab 1: Trích Xuất Hóa Đơn */}
        {activeTab === "extract" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* ── CỘT TRÁI: ĐẦU VÀO & HÌNH ẢNH (5 COLS) ── */}
            <div className="lg:col-span-5 space-y-4">
              {/* Card Chọn Mô Hình */}
              <div className="bg-[#111622] border border-white/10 rounded-2xl p-4 shadow-xl">
                <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                  Lựa chọn Mô hình AI
                </label>
                <div className="relative">
                  <select
                    value={model}
                    onChange={(e) => {
                      const newModel = e.target.value;
                      setModel(newModel);
                      if (file || selectedSampleId) {
                        executePredict(selectedSampleId, file, newModel);
                      }
                    }}
                    className="w-full appearance-none bg-[#182030] border border-white/15 rounded-xl px-3.5 py-2.5 text-white text-sm font-medium focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-all cursor-pointer"
                  >
                    {MODELS.map((m) => (
                      <option key={m.value} value={m.value}>
                        {m.name}
                      </option>
                    ))}
                  </select>
                  <div className="absolute right-3.5 top-1/2 -translate-y-1/2 pointer-events-none text-slate-400 text-xs">▼</div>
                </div>
                <p className="text-xs text-slate-400 mt-2">
                  {selectedModelInfo.desc}
                </p>
              </div>

              {/* Card Tải ảnh & Chọn mẫu */}
              <div className="bg-[#111622] border border-white/10 rounded-2xl p-4 shadow-xl">
                <div className="flex items-center justify-between mb-3">
                  <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                    Hình ảnh Hóa đơn
                  </label>
                  {preview && (
                    <button
                      onClick={() => {
                        setFile(null);
                        setSelectedSampleId(null);
                        setPreview(null);
                        setResult(null);
                      }}
                      className="text-xs text-rose-400 hover:underline"
                    >
                      Xóa ảnh
                    </button>
                  )}
                </div>

                {/* Image or Dropzone */}
                {!preview ? (
                  <div className="space-y-3">
                    <div
                      onDrop={handleDrop}
                      onDragOver={(e) => e.preventDefault()}
                      onClick={() => fileInputRef.current?.click()}
                      className="border-2 border-dashed border-white/15 hover:border-indigo-500/60 rounded-2xl p-6 text-center cursor-pointer transition-all bg-white/[0.01] hover:bg-indigo-500/[0.02]"
                    >
                      <UploadCloud className="w-10 h-10 text-indigo-400 mx-auto mb-2" />
                      <div className="font-semibold text-sm text-slate-200">Kéo thả ảnh hoặc bấm để chọn tệp</div>
                      <div className="text-xs text-slate-500 mt-1">Hỗ trợ JPG, PNG, WEBP (Hóa đơn mua sắm, GTGT, in nhiệt)</div>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <div className="relative border border-white/10 rounded-xl overflow-hidden bg-black flex items-center justify-center min-h-[300px] max-h-[420px]">
                      <div className="overflow-auto w-full h-full flex items-center justify-center p-2">
                        <img
                          src={preview}
                          alt="Hóa đơn"
                          style={{
                            transform: `scale(${zoomLevel}) rotate(${rotation}deg)`,
                            transition: "transform 0.2s ease",
                            maxWidth: "100%",
                            maxHeight: "400px",
                            objectFit: "contain"
                          }}
                          className="rounded"
                        />
                      </div>

                      {/* Floating Viewer Toolbar */}
                      <div className="absolute top-3 right-3 flex items-center gap-1.5 bg-black/70 backdrop-blur-md p-1.5 rounded-xl border border-white/10">
                        <button onClick={() => setZoomLevel((z) => Math.min(z + 0.25, 3))} title="Phóng to" className="p-1.5 hover:bg-white/10 rounded-lg text-slate-300">
                          <ZoomIn className="w-4 h-4" />
                        </button>
                        <button onClick={() => setZoomLevel((z) => Math.max(z - 0.25, 0.75))} title="Thu nhỏ" className="p-1.5 hover:bg-white/10 rounded-lg text-slate-300">
                          <ZoomOut className="w-4 h-4" />
                        </button>
                        <button onClick={() => setRotation((r) => (r + 90) % 360)} title="Xoay 90 độ" className="p-1.5 hover:bg-white/10 rounded-lg text-slate-300">
                          <RotateCw className="w-4 h-4" />
                        </button>
                        <button onClick={() => setIsFullscreen(true)} title="Xem toàn màn hình" className="p-1.5 hover:bg-white/10 rounded-lg text-slate-300">
                          <Maximize2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>

                    {/* Re-run button */}
                    <button
                      onClick={handlePredict}
                      disabled={loading}
                      className="w-full py-2.5 bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white rounded-xl text-xs font-bold shadow-lg shadow-indigo-600/30 flex items-center justify-center gap-2 transition-all disabled:opacity-50"
                    >
                      {loading ? (
                        <>
                          <Loader2 className="w-4 h-4 animate-spin" /> Đang trích xuất dữ liệu bằng AI...
                        </>
                      ) : (
                        <>
                          <RefreshCw className="w-3.5 h-3.5" /> Trích Xuất Lại (Chạy Lại AI)
                        </>
                      )}
                    </button>
                  </div>
                )}

                {/* Always-Visible Controls: Camera, Upload & 5 Presets */}
                <div className="pt-3 border-t border-white/10 space-y-2.5">
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      onClick={() => cameraInputRef.current?.click()}
                      className="w-full py-2 px-3 bg-indigo-600/80 hover:bg-indigo-600 text-white rounded-xl text-xs font-semibold flex items-center justify-center gap-1.5 transition-all shadow-md shadow-indigo-600/20"
                    >
                      <Camera className="w-3.5 h-3.5" /> Chụp Camera
                    </button>
                    <button
                      onClick={() => fileInputRef.current?.click()}
                      className="w-full py-2 px-3 bg-white/10 hover:bg-white/15 text-slate-200 rounded-xl text-xs font-semibold flex items-center justify-center gap-1.5 transition-all"
                    >
                      <FileImage className="w-3.5 h-3.5" /> Tải Ảnh Lên
                    </button>
                  </div>

                  {/* Hidden file inputs */}
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
                  />
                  <input
                    ref={cameraInputRef}
                    type="file"
                    accept="image/*"
                    capture="environment"
                    className="hidden"
                    onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
                  />

                  {/* 5 Preset Samples */}
                  <div>
                    <div className="text-[11px] font-medium text-slate-400 mb-1.5">Chọn mẫu hóa đơn có sẵn (1-click chạy ngay):</div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                      {PRESET_SAMPLES.map((s) => {
                        const Icon = s.icon;
                        const isSelected = selectedSampleId === s.id;
                        return (
                          <button
                            key={s.id}
                            onClick={() => handleSelectSample(s)}
                            className={`p-2 rounded-xl border text-left transition-all flex items-center gap-2 ${
                              isSelected
                                ? "border-indigo-500 bg-indigo-500/20 shadow-md shadow-indigo-500/10"
                                : "border-white/10 hover:border-indigo-500/50 bg-[#161d2c] hover:bg-[#1d273a]"
                            }`}
                          >
                            <div className={`p-1.5 rounded-lg ${isSelected ? "bg-indigo-500 text-white" : "bg-indigo-500/10 text-indigo-400"}`}>
                              <Icon className="w-3.5 h-3.5" />
                            </div>
                            <div className="overflow-hidden">
                              <div className={`font-semibold text-xs truncate ${isSelected ? "text-indigo-300" : "text-slate-200"}`}>{s.name}</div>
                              <div className="text-[10px] text-slate-500 truncate">{s.desc}</div>
                            </div>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* ── CỘT PHẢI: BẢNG DỮ LIỆU CHUẨN (7 COLS) ── */}
            <div className="lg:col-span-7">
              {!result && !loading && (
                <div className="h-full min-h-[400px] border border-dashed border-white/10 rounded-2xl flex flex-col items-center justify-center p-8 text-center bg-[#111622]/40">
                  <FileText className="w-14 h-14 text-slate-600 mb-3" />
                  <h3 className="font-semibold text-slate-300 text-base">Chưa có kết quả trích xuất</h3>
                  <p className="text-slate-500 text-xs max-w-sm mt-1">
                    Vui lòng chọn một hóa đơn mẫu bên trái hoặc tải ảnh hóa đơn của bạn lên, sau đó bấm &quot;Bắt Đầu Trích Xuất Dữ Liệu&quot;.
                  </p>
                </div>
              )}

              {loading && (
                <div className="h-full min-h-[400px] border border-white/10 rounded-2xl flex flex-col items-center justify-center p-8 text-center bg-[#111622]">
                  <Loader2 className="w-12 h-12 text-indigo-400 animate-spin mb-4" />
                  <h3 className="font-bold text-white text-base">AI Đang Xử Lý Hóa Đơn</h3>
                  <p className="text-slate-400 text-xs mt-1">
                    Mô hình đang đọc cấu trúc không gian và bóc tách dữ liệu vào định dạng bảng chuẩn...
                  </p>
                </div>
              )}

              {result && !loading && (
                <div className="bg-[#111622] border border-white/10 rounded-2xl overflow-hidden shadow-2xl space-y-5 p-5">
                  {/* Action Bar: Edit & Exports */}
                  <div className="flex flex-wrap items-center justify-between gap-2 pb-4 border-b border-white/10">
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-base text-white">Kết Quả Trích Xuất</span>
                      {latency && (
                        <span className="text-xs px-2.5 py-0.5 rounded-full bg-indigo-500/10 text-indigo-300 border border-indigo-500/20 font-mono">
                          {latency.toFixed(2)}s
                        </span>
                      )}
                    </div>

                    <div className="flex items-center gap-1.5">
                      <button
                        onClick={() => setIsEditing(!isEditing)}
                        className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all ${
                          isEditing
                            ? "bg-emerald-600 text-white shadow-sm"
                            : "bg-[#1d273a] hover:bg-[#25324a] text-slate-300 border border-white/10"
                        }`}
                      >
                        {isEditing ? <><Check className="w-3.5 h-3.5" /> Xong</> : <><Edit3 className="w-3.5 h-3.5" /> Chỉnh sửa</>}
                      </button>

                      <button
                        onClick={exportCSV}
                        className="px-3 py-1.5 bg-[#1d273a] hover:bg-[#25324a] text-slate-200 border border-white/10 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-all"
                      >
                        <Download className="w-3.5 h-3.5" /> Xuất Excel (CSV)
                      </button>

                      <button
                        onClick={copyJSONToClipboard}
                        className="px-3 py-1.5 bg-[#1d273a] hover:bg-[#25324a] text-slate-200 border border-white/10 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-all"
                      >
                        <Copy className="w-3.5 h-3.5" /> {copySuccess ? "Đã chép!" : "Chép JSON"}
                      </button>

                      <button
                        onClick={handlePrint}
                        className="px-3 py-1.5 bg-[#1d273a] hover:bg-[#25324a] text-slate-200 border border-white/10 rounded-lg text-xs font-medium hidden sm:flex items-center gap-1.5 transition-all"
                      >
                        <FileText className="w-3.5 h-3.5" /> In phiếu
                      </button>

                      {/* Lưu vào sổ hóa đơn button */}
                      <button
                        onClick={handleSaveReceipt}
                        disabled={isSaving}
                        className="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all shadow-md shadow-emerald-600/20 disabled:opacity-50"
                        title="Lưu kết quả hóa đơn vào Sổ Kế Toán"
                      >
                        {isSaving ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Save className="w-3.5 h-3.5" />}
                        Lưu Sổ Hóa Đơn
                      </button>
                    </div>
                  </div>

                  {/* Save success toast banner */}
                  {saveSuccessMsg && (
                    <div className="p-3 rounded-xl bg-emerald-500/15 border border-emerald-500/40 text-emerald-300 text-xs flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <CheckCheck className="w-4 h-4 text-emerald-400 shrink-0" />
                        <span className="font-semibold">{saveSuccessMsg}</span>
                      </div>
                      <button
                        onClick={() => {
                          setActiveTab("history");
                          fetchHistory();
                        }}
                        className="text-[11px] underline hover:text-white font-medium"
                      >
                        Mở Sổ Lưu Trữ →
                      </button>
                    </div>
                  )}

                  {/* 1. THẺ THÔNG TIN CHUNG (4 Header Fields) */}
                  <div>
                    <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2.5">
                      1. Thông Tin Chung Hóa Đơn
                    </h4>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      <div className="bg-[#161d2c] border border-white/5 rounded-xl p-3">
                        <span className="text-[11px] text-slate-400 block mb-1">Đơn vị bán hàng (SELLER):</span>
                        {isEditing ? (
                          <input
                            type="text"
                            value={editableData.SELLER || ""}
                            onChange={(e) => handleFieldChange("SELLER", e.target.value)}
                            className="w-full bg-[#0d121d] border border-indigo-500/50 rounded px-2 py-1 text-xs text-white"
                          />
                        ) : (
                          <span className="font-semibold text-sm text-white">{editableData.SELLER || "—"}</span>
                        )}
                      </div>

                      <div className="bg-[#161d2c] border border-white/5 rounded-xl p-3">
                        <span className="text-[11px] text-slate-400 block mb-1">Thời gian lập (TIMESTAMP):</span>
                        {isEditing ? (
                          <input
                            type="text"
                            value={editableData.TIMESTAMP || ""}
                            onChange={(e) => handleFieldChange("TIMESTAMP", e.target.value)}
                            className="w-full bg-[#0d121d] border border-indigo-500/50 rounded px-2 py-1 text-xs text-white"
                          />
                        ) : (
                          <span className="font-semibold text-sm text-white">{editableData.TIMESTAMP || "—"}</span>
                        )}
                      </div>

                      <div className="sm:col-span-2 bg-[#161d2c] border border-white/5 rounded-xl p-3">
                        <span className="text-[11px] text-slate-400 block mb-1">Địa chỉ (ADDRESS):</span>
                        {isEditing ? (
                          <input
                            type="text"
                            value={editableData.ADDRESS || ""}
                            onChange={(e) => handleFieldChange("ADDRESS", e.target.value)}
                            className="w-full bg-[#0d121d] border border-indigo-500/50 rounded px-2 py-1 text-xs text-white"
                          />
                        ) : (
                          <span className="text-xs text-slate-200 leading-relaxed">{editableData.ADDRESS || "—"}</span>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* 2. BẢNG CHI TIẾT MẶT HÀNG (Normal Clean Table) */}
                  <div>
                    <div className="flex items-center justify-between mb-2.5">
                      <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider">
                        2. Bảng Danh Sách Hàng Hóa ({items.length} mặt hàng)
                      </h4>
                      {isEditing && (
                        <button
                          onClick={handleAddItem}
                          className="px-2.5 py-1 bg-indigo-600/30 text-indigo-300 border border-indigo-500/30 rounded-lg text-xs flex items-center gap-1 hover:bg-indigo-600/50"
                        >
                          <Plus className="w-3.5 h-3.5" /> Thêm dòng
                        </button>
                      )}
                    </div>

                    <div className="border border-white/10 rounded-xl overflow-hidden bg-[#141b29]">
                      <div className="overflow-x-auto">
                        <table className="w-full text-left text-xs">
                          <thead className="bg-[#1a2335] text-slate-300 border-b border-white/10">
                            <tr>
                              <th className="py-2.5 px-3 font-semibold text-center w-10">STT</th>
                              <th className="py-2.5 px-3 font-semibold">Tên mặt hàng</th>
                              <th className="py-2.5 px-3 font-semibold text-center w-16">SL</th>
                              <th className="py-2.5 px-3 font-semibold text-right w-28">Đơn giá (đ)</th>
                              <th className="py-2.5 px-3 font-semibold text-right w-28">Thành tiền (đ)</th>
                              {isEditing && <th className="py-2.5 px-3 text-center w-10">Xóa</th>}
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-white/5 text-slate-200">
                            {items.length === 0 ? (
                              <tr>
                                <td colSpan={isEditing ? 6 : 5} className="py-6 text-center text-slate-500 italic">
                                  Không tìm thấy chi tiết mặt hàng
                                </td>
                              </tr>
                            ) : (
                              items.map((it, idx) => (
                                <tr key={idx} className="hover:bg-white/[0.03] transition-colors">
                                  <td className="py-2.5 px-3 text-center text-slate-500 font-mono">{idx + 1}</td>
                                  <td className="py-2.5 px-3">
                                    {isEditing ? (
                                      <input
                                        type="text"
                                        value={it.name}
                                        onChange={(e) => handleItemChange(idx, "name", e.target.value)}
                                        className="w-full bg-[#0d121d] border border-indigo-500/50 rounded px-2 py-1 text-xs text-white"
                                      />
                                    ) : (
                                      <span className="font-medium text-white">{it.name || "—"}</span>
                                    )}
                                  </td>
                                  <td className="py-2.5 px-3 text-center">
                                    {isEditing ? (
                                      <input
                                        type="text"
                                        value={it.qty}
                                        onChange={(e) => handleItemChange(idx, "qty", e.target.value)}
                                        className="w-full bg-[#0d121d] border border-indigo-500/50 rounded px-1.5 py-1 text-xs text-center text-white"
                                      />
                                    ) : (
                                      it.qty || "1"
                                    )}
                                  </td>
                                  <td className="py-2.5 px-3 text-right text-slate-400 font-mono">
                                    {isEditing ? (
                                      <input
                                        type="text"
                                        value={it.price}
                                        onChange={(e) => handleItemChange(idx, "price", e.target.value)}
                                        className="w-full bg-[#0d121d] border border-indigo-500/50 rounded px-1.5 py-1 text-xs text-right text-white"
                                      />
                                    ) : (
                                      it.price || <span className="text-slate-600 italic">khuyết</span>
                                    )}
                                  </td>
                                  <td className="py-2.5 px-3 text-right font-semibold text-white font-mono">
                                    {isEditing ? (
                                      <input
                                        type="text"
                                        value={it.amount}
                                        onChange={(e) => handleItemChange(idx, "amount", e.target.value)}
                                        className="w-full bg-[#0d121d] border border-indigo-500/50 rounded px-1.5 py-1 text-xs text-right text-white"
                                      />
                                    ) : (
                                      it.amount ? `${it.amount} đ` : "—"
                                    )}
                                  </td>
                                  {isEditing && (
                                    <td className="py-2.5 px-3 text-center">
                                      <button
                                        onClick={() => handleDeleteItem(idx)}
                                        className="p-1 hover:bg-rose-500/20 text-rose-400 rounded transition-all"
                                        title="Xóa hàng này"
                                      >
                                        <Trash2 className="w-3.5 h-3.5" />
                                      </button>
                                    </td>
                                  )}
                                </tr>
                              ))
                            )}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </div>

                  {/* 3. TỔNG THANH TOÁN & ĐỐI SOÁT SỐ HỌC */}
                  <div className="bg-[#161d2c] border border-white/10 rounded-xl p-4 space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-bold text-slate-300 uppercase">
                        Tổng Tiền Thanh Toán (TOTAL_COST):
                      </span>
                      {isEditing ? (
                        <input
                          type="text"
                          value={editableData.TOTAL_COST || ""}
                          onChange={(e) => handleFieldChange("TOTAL_COST", e.target.value)}
                          className="bg-[#0d121d] border border-emerald-500/50 rounded px-3 py-1 text-base font-bold text-emerald-400 text-right w-44"
                        />
                      ) : (
                        <span className="text-xl font-black text-emerald-400 font-mono">
                          {editableData.TOTAL_COST || "0"} đ
                        </span>
                      )}
                    </div>

                    {/* Live Arithmetic Check Badge */}
                    <div className={`p-2.5 rounded-lg text-xs flex items-center gap-2 border ${
                      isMatch
                        ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-300"
                        : "bg-amber-500/10 border-amber-500/30 text-amber-300"
                    }`}>
                      {isMatch ? (
                        <>
                          <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0" />
                          <span>✓ Khớp số học 100%: Tổng các món ({itemsSum.toLocaleString()} đ) khớp chính xác với Tổng thanh toán ({declaredTotal.toLocaleString()} đ)</span>
                        </>
                      ) : (
                        <>
                          <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
                          <span>
                            Cảnh báo sai lệch số học: Tổng các món ({itemsSum.toLocaleString()} đ) lệch {(itemsSum - declaredTotal > 0 ? "+" : "") + (itemsSum - declaredTotal).toLocaleString()} đ so với Tổng thanh toán ({declaredTotal.toLocaleString()} đ)
                          </span>
                        </>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Tab 2: So Sánh 2 Mô Hình */}
        {activeTab === "compare" && (
          <div className="bg-[#111622] border border-white/10 rounded-2xl p-6 shadow-xl space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-white/10 pb-4">
              <div>
                <h3 className="text-lg font-bold text-white">So Sánh Đối Đầu Giữa 2 Mô Hình</h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Chạy cùng một ảnh hóa đơn qua 2 kiến trúc khác nhau để đối chiếu trực tiếp chất lượng trích xuất.
                </p>
              </div>

              <div className="flex items-center gap-3">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-slate-400">Đối sánh với:</span>
                  <select
                    value={compareModel}
                    onChange={(e) => setCompareModel(e.target.value)}
                    className="bg-[#182030] border border-white/15 rounded-lg px-3 py-1.5 text-xs text-white"
                  >
                    {MODELS.filter((m) => m.value !== model).map((m) => (
                      <option key={m.value} value={m.value}>{m.name}</option>
                    ))}
                  </select>
                </div>
                <button
                  onClick={handleCompare}
                  disabled={compareLoading || (!file && !selectedSampleId)}
                  className="px-4 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold rounded-lg disabled:opacity-50 flex items-center gap-1.5"
                >
                  {compareLoading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5 fill-white" />}
                  Chạy So Sánh
                </button>
              </div>
            </div>

            {!preview && (
              <div className="text-center py-12 text-slate-500 text-xs italic">
                Vui lòng quay lại tab &quot;Trích Xuất Hóa Đơn&quot; để chọn hoặc tải ảnh lên trước.
              </div>
            )}

            {compareResult && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Left Model */}
                <div className="border border-indigo-500/30 rounded-xl p-4 bg-[#141a27] space-y-3">
                  <div className="flex items-center justify-between border-b border-white/10 pb-2">
                    <span className="font-bold text-indigo-400 text-sm">{selectedModelInfo.name}</span>
                    <span className="text-xs text-slate-400 font-mono">{compareResult.left.latency_seconds?.toFixed(2)}s</span>
                  </div>
                  <div className="text-xs space-y-1.5">
                    <div><strong className="text-slate-400">Đơn vị:</strong> {compareResult.left.extraction?.SELLER || "—"}</div>
                    <div><strong className="text-slate-400">Tổng tiền:</strong> <span className="text-emerald-400 font-bold">{compareResult.left.extraction?.TOTAL_COST || "—"} đ</span></div>
                    <div><strong className="text-slate-400">Số món trích được:</strong> {compareResult.left.extraction?.ITEMS?.length || 0} món</div>
                  </div>
                </div>

                {/* Right Model */}
                <div className="border border-white/10 rounded-xl p-4 bg-[#141a27] space-y-3">
                  <div className="flex items-center justify-between border-b border-white/10 pb-2">
                    <span className="font-bold text-slate-300 text-sm">
                      {MODELS.find((m) => m.value === compareModel)?.name}
                    </span>
                    <span className="text-xs text-slate-400 font-mono">{compareResult.right.latency_seconds?.toFixed(2)}s</span>
                  </div>
                  <div className="text-xs space-y-1.5">
                    <div><strong className="text-slate-400">Đơn vị:</strong> {compareResult.right.extraction?.SELLER || "—"}</div>
                    <div><strong className="text-slate-400">Tổng tiền:</strong> <span className="text-emerald-400 font-bold">{compareResult.right.extraction?.TOTAL_COST || "—"} đ</span></div>
                    <div><strong className="text-slate-400">Số món trích được:</strong> {compareResult.right.extraction?.ITEMS?.length || 0} món</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Tab 3: Hỏi Đáp với Hóa Đơn */}
        {activeTab === "chat" && (
          <div className="bg-[#111622] border border-white/10 rounded-2xl p-6 shadow-xl space-y-4 max-w-4xl mx-auto">
            <div>
              <h3 className="text-lg font-bold text-white">Hỏi Đáp Trực Tiếp Về Nội Dung Hóa Đơn (Q&A)</h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Đặt câu hỏi tự nhiên bằng tiếng Việt để AI tra cứu thông tin từ dữ liệu hóa đơn vừa trích xuất.
              </p>
            </div>

            {/* Quick suggested questions */}
            <div className="flex flex-wrap gap-2 pt-2">
              {SUGGESTED_QUESTIONS.map((sq) => (
                <button
                  key={sq}
                  onClick={() => setChatInput(sq)}
                  className="text-xs px-3 py-1.5 rounded-lg bg-[#182030] hover:bg-indigo-600/30 text-slate-300 border border-white/10 hover:border-indigo-500/40 transition-all"
                >
                  {sq}
                </button>
              ))}
            </div>

            {/* Chat Box */}
            <div className="border border-white/10 rounded-xl p-4 bg-[#0d121d] min-h-[300px] max-h-[420px] overflow-y-auto space-y-3">
              {messages.length === 0 ? (
                <div className="text-center py-14 text-slate-500 text-xs italic">
                  Chưa có tin nhắn nào. Hãy chọn hoặc nhập câu hỏi bên dưới để bắt đầu.
                </div>
              ) : (
                messages.map((m, idx) => (
                  <div key={idx} className={`flex gap-2.5 ${m.role === "user" ? "justify-end" : "justify-start"}`}>
                    {m.role === "assistant" && (
                      <div className="w-7 h-7 rounded-lg bg-indigo-600 flex items-center justify-center shrink-0">
                        <Bot className="w-4 h-4 text-white" />
                      </div>
                    )}
                    <div className={`p-3 rounded-xl text-xs max-w-lg leading-relaxed ${
                      m.role === "user"
                        ? "bg-indigo-600 text-white rounded-tr-none"
                        : "bg-[#182030] text-slate-200 border border-white/10 rounded-tl-none"
                    }`}>
                      {m.content}
                    </div>
                    {m.role === "user" && (
                      <div className="w-7 h-7 rounded-lg bg-slate-700 flex items-center justify-center shrink-0">
                        <User className="w-4 h-4 text-white" />
                      </div>
                    )}
                  </div>
                ))
              )}
              {chatLoading && (
                <div className="flex items-center gap-2 text-xs text-slate-400">
                  <Loader2 className="w-4 h-4 animate-spin text-indigo-400" /> AI đang trả lời...
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            {/* Chat Input */}
            <div className="flex gap-2">
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleChat()}
                placeholder="Nhập câu hỏi về hóa đơn (ví dụ: Tổng tiền là bao nhiêu?)..."
                className="flex-1 bg-[#182030] border border-white/15 rounded-xl px-4 py-2.5 text-xs text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
              <button
                onClick={handleChat}
                disabled={chatLoading || !chatInput.trim()}
                className="px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-xs font-bold disabled:opacity-50 flex items-center gap-1.5 transition-all shadow-md shadow-indigo-600/20"
              >
                <Send className="w-3.5 h-3.5" /> Gửi
              </button>
            </div>
          </div>
        )}

        {/* Tab 4: Sổ Lưu Trữ Hóa Đơn (Audit Ledger) */}
        {activeTab === "history" && (
          <div className="space-y-6">
            {/* Header & KPI Summary Cards */}
            <div className="bg-[#111622] border border-white/10 rounded-2xl p-6 shadow-xl space-y-6">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-white/10 pb-4">
                <div>
                  <h3 className="text-lg font-bold text-white flex items-center gap-2">
                    <Archive className="w-5 h-5 text-indigo-400" /> Sổ Lưu Trữ & Đối Soát Hóa Đơn (Audit Ledger)
                  </h3>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Toàn bộ hóa đơn đã được AI bóc tách và kế toán viên kiểm duyệt, lưu trữ phục vụ nghiệp vụ doanh nghiệp.
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <a
                    href={`${API_BASE}/api/history/export_csv`}
                    download="so_ke_toan_hoa_don.csv"
                    className="px-3.5 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl text-xs font-semibold flex items-center gap-1.5 transition-all shadow-md shadow-emerald-600/20"
                  >
                    <Download className="w-4 h-4" /> Xuất Toàn Bộ Sổ (CSV / Excel)
                  </a>
                  <button
                    onClick={fetchHistory}
                    disabled={historyLoading}
                    className="px-3 py-2 bg-[#182030] hover:bg-[#202b40] text-slate-300 border border-white/10 rounded-xl text-xs font-medium flex items-center gap-1.5 transition-all"
                    title="Làm mới danh sách"
                  >
                    <RefreshCw className={`w-3.5 h-3.5 ${historyLoading ? "animate-spin" : ""}`} />
                  </button>
                </div>
              </div>

              {/* 3 Metric KPI Cards */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="bg-[#161d2c] border border-white/10 rounded-xl p-4">
                  <span className="text-xs font-medium text-slate-400 block">Tổng Hóa Đơn Đã Lưu</span>
                  <div className="text-2xl font-black text-white mt-1">
                    {historySummary?.total_receipts || historyRecords.length} <span className="text-xs font-normal text-slate-400">chứng từ</span>
                  </div>
                </div>
                <div className="bg-[#161d2c] border border-white/10 rounded-xl p-4">
                  <span className="text-xs font-medium text-slate-400 block">Tổng Doanh Số Ghi Nhận</span>
                  <div className="text-2xl font-black text-emerald-400 mt-1 font-mono">
                    {(historySummary?.total_revenue || 0).toLocaleString()} <span className="text-xs font-normal text-slate-400">đ</span>
                  </div>
                </div>
                <div className="bg-[#161d2c] border border-white/10 rounded-xl p-4">
                  <span className="text-xs font-medium text-slate-400 block">Tỷ Lệ Khớp Số Học</span>
                  <div className="text-2xl font-black text-indigo-400 mt-1">
                    {historySummary?.verified_rate || 100}% <span className="text-xs font-normal text-slate-400">chính xác</span>
                  </div>
                </div>
              </div>

              {/* Search Bar */}
              <div className="relative">
                <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
                <input
                  type="text"
                  value={historySearch}
                  onChange={(e) => setHistorySearch(e.target.value)}
                  placeholder="Tìm kiếm theo tên đơn vị bán hàng, địa chỉ, hoặc mã chứng từ..."
                  className="w-full bg-[#161d2c] border border-white/10 rounded-xl pl-10 pr-4 py-2.5 text-xs text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>

              {/* History Table */}
              <div className="border border-white/10 rounded-xl overflow-hidden bg-[#0d121d]">
                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse text-xs">
                    <thead>
                      <tr className="border-b border-white/10 bg-[#161d2c] text-slate-400 uppercase text-[11px] font-semibold">
                        <th className="py-3 px-3.5">Mã Chứng Từ</th>
                        <th className="py-3 px-3.5">Thời Gian Lưu</th>
                        <th className="py-3 px-3.5">Đơn Vị Bán Hàng</th>
                        <th className="py-3 px-3.5 text-center">Số Món</th>
                        <th className="py-3 px-3.5 text-right">Tổng Tiền (VNĐ)</th>
                        <th className="py-3 px-3.5 text-center">Đối Soát Số Học</th>
                        <th className="py-3 px-3.5 text-center">Thao Tác</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5">
                      {historyRecords
                        .filter((r) => {
                          if (!historySearch.trim()) return true;
                          const q = historySearch.toLowerCase();
                          return (
                            (r.seller || "").toLowerCase().includes(q) ||
                            (r.address || "").toLowerCase().includes(q) ||
                            (r.id || "").toLowerCase().includes(q)
                          );
                        })
                        .map((rec) => (
                          <tr key={rec.id} className="hover:bg-white/[0.02] transition-colors">
                            <td className="py-3 px-3.5 font-mono text-indigo-400 font-semibold">{rec.id}</td>
                            <td className="py-3 px-3.5 text-slate-400">{rec.created_at}</td>
                            <td className="py-3 px-3.5 text-white font-medium">
                              <div>{rec.seller}</div>
                              {rec.address && <div className="text-[11px] text-slate-500 truncate max-w-xs">{rec.address}</div>}
                            </td>
                            <td className="py-3 px-3.5 text-center text-slate-300 font-semibold">{rec.item_count} món</td>
                            <td className="py-3 px-3.5 text-right font-mono font-bold text-emerald-400">
                              {rec.total_amount ? rec.total_amount.toLocaleString() : rec.total_cost} đ
                            </td>
                            <td className="py-3 px-3.5 text-center">
                              {rec.is_valid ? (
                                <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                                  <CheckCircle className="w-3 h-3" /> Khớp 100%
                                </span>
                              ) : (
                                <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/30">
                                  <AlertTriangle className="w-3 h-3" /> Lệch số học
                                </span>
                              )}
                            </td>
                            <td className="py-3 px-3.5 text-center">
                              <div className="flex items-center justify-center gap-1.5">
                                <button
                                  onClick={() => handleLoadFromHistory(rec)}
                                  className="px-2.5 py-1 bg-indigo-600/30 hover:bg-indigo-600 text-indigo-300 hover:text-white rounded-lg text-xs font-medium flex items-center gap-1 transition-all"
                                  title="Xem lại trên bảng trích xuất"
                                >
                                  <Eye className="w-3.5 h-3.5" /> Xem lại
                                </button>
                                <button
                                  onClick={() => handleDeleteHistory(rec.id)}
                                  className="p-1 hover:bg-rose-500/20 text-rose-400 rounded-lg transition-all"
                                  title="Xóa khỏi sổ"
                                >
                                  <Trash2 className="w-3.5 h-3.5" />
                                </button>
                              </div>
                            </td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* ── LIGHTBOX FULLSCREEN MODAL ── */}
      {isFullscreen && preview && (
        <div
          className="fixed inset-0 z-50 bg-black/90 backdrop-blur-md flex items-center justify-center p-4"
          onClick={() => setIsFullscreen(false)}
        >
          <button
            onClick={() => setIsFullscreen(false)}
            className="absolute top-5 right-5 p-2 bg-white/10 hover:bg-white/20 text-white rounded-full transition-all"
          >
            <X className="w-6 h-6" />
          </button>
          <img
            src={preview}
            alt="Toàn màn hình"
            className="max-w-full max-h-[90vh] object-contain rounded-lg shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}
    </div>
  );
}
