import type { NextConfig } from "next";

const BACKEND_URL = (
  process.env.NEXT_PUBLIC_BACKEND_URL ||
  process.env.BACKEND_URL ||
  "http://127.0.0.1:8000"
).replace(/\/$/, "");

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${BACKEND_URL}/api/:path*`,
      },
      {
        source: "/templates_images/:path*",
        destination: `${BACKEND_URL}/templates_images/:path*`,
      },
      {
        source: "/temp_uploads/:path*",
        destination: `${BACKEND_URL}/temp_uploads/:path*`,
      },
    ];
  },
};

export default nextConfig;
