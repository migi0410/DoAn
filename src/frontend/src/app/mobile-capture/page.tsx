'use client';

import React, { useState, useRef } from 'react';
import { Camera, UploadCloud, Loader2, CheckCircle, RefreshCcw } from 'lucide-react';
import { supabase } from '@/lib/supabaseClient';

const TEMPLATES = [
  "cafe_highlands", "cafe_phuclong", "cafe_starbucks",
  "convenience_7eleven", "convenience_circlek", "convenience_gs25",
  "einvoice_viettel", "einvoice_vnpt", "minimart_anan", "receipt_c45_bb",
  "restaurant_jollibee", "restaurant_kfc", "supermarket_bachhoaxanh",
  "supermarket_lotte", "supermarket_winmart"
];

export default function MobileCapturePage() {
  const [template, setTemplate] = useState(TEMPLATES[0]);
  const [uploading, setUploading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [uploadingCount, setUploadingCount] = useState(0);
  const [totalUpload, setTotalUpload] = useState(0);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    try {
      if (!e.target.files || e.target.files.length === 0) {
        return;
      }
      
      const selectedFiles = Array.from(e.target.files);
      setTotalUpload(selectedFiles.length);
      setUploadingCount(0);
      setSuccess(false);
      setErrorMessage(null);
      
      // Auto-upload immediately
      await handleBatchUpload(selectedFiles);
    } catch (err: any) {
      alert("Lỗi đọc file: " + err.message);
    }
  };

  const handleBatchUpload = async (files: File[]) => {
    setUploading(true);
    setErrorMessage(null);
    let successCount = 0;
    
    try {
      for (const uploadFile of files) {
        // Create a unique filename: template_timestamp_index.jpg
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const randomId = Math.random().toString(36).substring(7);
        const ext = uploadFile.name.split('.').pop() || 'jpg';
        const fileName = `${template}/${template}_${timestamp}_${randomId}.${ext}`;

        const { error } = await supabase.storage
          .from('raw_images')
          .upload(fileName, uploadFile, {
            cacheControl: '3600',
            upsert: false
          });

        if (error) {
          console.error("Upload error for file", uploadFile.name, error);
          // Vẫn tiếp tục upload các file khác, nhưng có thể báo lỗi sau
        } else {
          successCount++;
          setUploadingCount(successCount);
        }
      }

      if (successCount === 0 && files.length > 0) {
        setErrorMessage("Lỗi: Không file nào upload thành công. Hãy kiểm tra lại RLS Policy.");
      } else if (successCount < files.length) {
        setErrorMessage(`Cảnh báo: Chỉ tải lên được ${successCount}/${files.length} ảnh.`);
        setSuccess(true);
      } else {
        setSuccess(true);
      }
    } catch (err: any) {
      console.error("Unknown error:", err);
      setErrorMessage(`Lỗi hệ thống: ${err.message}`);
    } finally {
      setUploading(false);
    }
  };

  const resetForm = () => {
    setSuccess(false);
    setErrorMessage(null);
    setTotalUpload(0);
    setUploadingCount(0);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 p-4 font-sans selection:bg-indigo-500/30">
      <div className="max-w-md mx-auto">
        <div className="text-center mb-8 mt-4">
          <h1 className="text-2xl font-bold bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">
            Thu Thập Hóa Đơn
          </h1>
          <p className="text-slate-500 text-sm mt-1">Chụp và đẩy hàng loạt lên Supabase Storage</p>
        </div>

        {/* Template Selection */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-slate-400 mb-2">Chọn Template</label>
          <select 
            value={template} 
            onChange={(e) => setTemplate(e.target.value)}
            className="w-full bg-slate-900 border border-slate-800 rounded-xl px-4 py-3 text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 appearance-none"
          >
            {TEMPLATES.map(t => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </div>

        {/* Camera/Preview Area */}
        <div className="mb-6">
          {uploading ? (
            <div className="relative rounded-2xl overflow-hidden border border-slate-800 bg-slate-900 aspect-square flex flex-col items-center justify-center text-center p-6">
               <Loader2 className="w-12 h-12 animate-spin text-indigo-500 mb-4" />
               <p className="font-bold text-lg text-white">Đang đẩy lên Cloud...</p>
               <p className="text-indigo-300 mt-2 font-medium">Hoàn thành: {uploadingCount} / {totalUpload}</p>
               <div className="w-full bg-slate-800 rounded-full h-2.5 mt-6 overflow-hidden">
                 <div className="bg-indigo-500 h-2.5 rounded-full transition-all duration-300" style={{ width: `${(uploadingCount / totalUpload) * 100}%` }}></div>
               </div>
            </div>
          ) : (
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 aspect-square flex flex-col items-center justify-center text-center">
              <div className="bg-slate-800 p-4 rounded-full mb-6">
                <Camera className="w-10 h-10 text-indigo-400" />
              </div>
              <h3 className="text-lg font-bold text-slate-200 mb-2">Tải ảnh hàng loạt</h3>
              <p className="text-sm text-slate-500 mb-6">Bạn có thể quét chọn nhiều ảnh cùng lúc</p>
              
              <input 
                type="file" 
                accept="image/*"
                multiple
                onClick={(e) => { (e.target as HTMLInputElement).value = ''; }}
                onChange={handleFileChange}
                className="block w-full max-w-xs mx-auto text-sm text-slate-400
                  file:mr-4 file:py-3 file:px-6
                  file:rounded-full file:border-0
                  file:text-sm file:font-semibold
                  file:bg-indigo-500 file:text-white
                  hover:file:bg-indigo-600 file:cursor-pointer
                  file:transition-colors file:shadow-lg"
              />
            </div>
          )}
        </div>

        {/* Success State */}
        {success && (
          <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-xl p-4 flex flex-col items-center justify-center text-center mb-6">
            <CheckCircle className="w-12 h-12 text-emerald-400 mb-2" />
            <h3 className="text-lg font-bold text-emerald-400">Đã tải lên {uploadingCount} ảnh!</h3>
            <p className="text-sm text-emerald-400/70 mb-4">Lưu thành công vào thư mục {template}</p>
            
            <button 
              onClick={resetForm}
              className="w-full py-3 bg-emerald-500 text-white rounded-lg font-semibold active:scale-[0.98] transition-transform"
            >
              Tiếp tục Tải lên
            </button>
          </div>
        )}

        {/* Error State */}
        {errorMessage && (
          <div className="bg-rose-500/10 border border-rose-500/20 rounded-xl p-4 flex flex-col items-center justify-center text-center mt-4">
            <h3 className="text-lg font-bold text-rose-400">Tải lên thất bại</h3>
            <p className="text-sm text-rose-400/80 mb-2">{errorMessage}</p>
            <p className="text-xs text-rose-300/60">Gợi ý: Kiểm tra lại Storage Policies của Supabase xem đã cấp quyền INSERT chưa.</p>
          </div>
        )}

      </div>
    </div>
  );
}
