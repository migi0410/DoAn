import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://127.0.0.1:8000/api/:path*",
      },
      {
        source: "/templates_images/:path*",
        destination: "http://127.0.0.1:8000/templates_images/:path*",
      },
      {
        source: "/temp_uploads/:path*",
        destination: "http://127.0.0.1:8000/temp_uploads/:path*",
      },
    ];
  },
};

export default nextConfig;
