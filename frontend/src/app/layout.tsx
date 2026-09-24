import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AVIR-KIE · AI Trích Xuất Hóa Đơn Tiếng Việt (Qwen3-VL LoRA v2)",
  description: "Hệ thống trích xuất thông tin hóa đơn tiếng Việt End-to-End ứng dụng Vision-Language Model Qwen3-VL 8B LoRA v2 và GPU PopOS.",
  icons: {
    icon: "/favicon.ico",
  }
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
  themeColor: "#ffffff",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="vi" className="h-full antialiased">
      <body className="min-h-full flex flex-col bg-slate-50 text-slate-900 font-sans selection:bg-indigo-100 selection:text-indigo-900">
        {children}
      </body>
    </html>
  );
}
