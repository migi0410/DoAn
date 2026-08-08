"use client";

import React, { useState } from "react";
import { useConfig } from "@/context/ConfigContext";
import { Settings, Check, Activity, X } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import axios from "axios";

export const Navbar = () => {
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  
  return (
    <>
      <nav className="w-full bg-white/80 backdrop-blur-md border-b border-slate-200 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 items-center">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-blue-500/20">
                <span className="text-white font-bold text-lg">A</span>
              </div>
              <span className="font-semibold text-lg tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-indigo-600">
                AVIR-KIE
              </span>
            </div>
            
            <button
              onClick={() => setIsSettingsOpen(true)}
              className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white hover:bg-slate-50 border border-slate-200 shadow-sm transition-colors text-sm text-slate-700"
            >
              <Settings className="w-4 h-4 text-slate-500" />
              <span className="font-medium">Config API</span>
            </button>
          </div>
        </div>
      </nav>

      <SettingsModal isOpen={isSettingsOpen} onClose={() => setIsSettingsOpen(false)} />
    </>
  );
};

const SettingsModal = ({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) => {
  const { apiUrl, setApiUrl } = useConfig();
  const [tempUrl, setTempUrl] = useState(apiUrl);
  const [isTesting, setIsTesting] = useState(false);
  const [testResult, setTestResult] = useState<"none" | "success" | "error">("none");

  // Sync state when opening
  React.useEffect(() => {
    if (isOpen) setTempUrl(apiUrl);
  }, [isOpen, apiUrl]);

  const handleSave = () => {
    setApiUrl(tempUrl);
    onClose();
  };

  const handleTest = async () => {
    setIsTesting(true);
    setTestResult("none");
    try {
      const res = await axios.get(`${tempUrl}/`);
      if (res.data?.status === "ok") {
        setTestResult("success");
      } else {
        setTestResult("error");
      }
    } catch (error) {
      setTestResult("error");
    } finally {
      setIsTesting(false);
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="absolute inset-0 bg-slate-900/20 backdrop-blur-sm"
          />
          <motion.div 
            initial={{ opacity: 0, scale: 0.95, y: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 10 }}
            className="relative w-full max-w-md bg-white border border-slate-200 rounded-2xl shadow-xl overflow-hidden"
          >
            <div className="flex justify-between items-center p-5 border-b border-slate-100">
              <h3 className="text-lg font-semibold text-slate-900 flex items-center gap-2">
                <Settings className="w-5 h-5 text-blue-500" /> API Configuration
              </h3>
              <button onClick={onClose} className="text-slate-400 hover:text-slate-600 transition-colors bg-slate-100 hover:bg-slate-200 p-1 rounded-full">
                <X className="w-4 h-4" />
              </button>
            </div>
            
            <div className="p-6 space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-semibold text-slate-700">Backend API URL</label>
                <p className="text-xs text-slate-500 mb-2">
                  Enter the URL where your FastAPI backend is running (e.g. Localhost or RunPod URL)
                </p>
                <div className="relative">
                  <input
                    type="text"
                    value={tempUrl}
                    onChange={(e) => setTempUrl(e.target.value)}
                    placeholder="https://xxx-runpod.net"
                    className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-2.5 text-sm text-slate-900 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all shadow-inner"
                  />
                </div>
              </div>

              <div className="flex items-center gap-3 pt-2">
                <button
                  onClick={handleTest}
                  disabled={isTesting}
                  className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg text-sm font-semibold transition-colors disabled:opacity-50 border border-slate-200"
                >
                  {isTesting ? (
                    <div className="w-4 h-4 border-2 border-slate-400 border-t-transparent rounded-full animate-spin" />
                  ) : (
                    <Activity className="w-4 h-4 text-slate-500" />
                  )}
                  Test Connection
                </button>
                <button
                  onClick={handleSave}
                  className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-semibold transition-colors shadow-md shadow-blue-500/20"
                >
                  <Check className="w-4 h-4" />
                  Save & Apply
                </button>
              </div>

              {testResult === "success" && (
                <div className="p-3 bg-green-50 border border-green-200 rounded-lg mt-4 shadow-sm">
                  <p className="text-sm text-green-700 font-medium flex items-center gap-2">
                    <Check className="w-4 h-4 text-green-600" /> Connection successful!
                  </p>
                </div>
              )}
              {testResult === "error" && (
                <div className="p-3 bg-red-50 border border-red-200 rounded-lg mt-4 shadow-sm">
                  <p className="text-sm text-red-700 font-medium flex items-center gap-2">
                    <X className="w-4 h-4 text-red-600" /> Connection failed. Check URL.
                  </p>
                </div>
              )}
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};
