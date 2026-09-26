import type { NextConfig } from "next";

const BACKEND_URL = (
  process.env.NEXT_PUBLIC_BACKEND_URL ||
  process.env.BACKEND_URL ||
  "https://kie.tavitax.com"
).replace(/\/$/, "");

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${BACKEND_URL}/api/:path*`,
      },
      {
        source: "/temp_uploads/:path*",
        destination: `${BACKEND_URL}/temp_uploads/:path*`,
      },
    ];
  },
};

export default nextConfig;
