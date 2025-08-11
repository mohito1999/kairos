import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'res.cloudinary.com',
        port: '',
        pathname: '/**',
      },
      {
        protocol: 'https',
        hostname: 'api.dicebear.com',
        port: '',
        pathname: '/**',
      },
    ],
  },
  
  // --- THE CORRECT FIX ---
  // Nesting the Server Actions configuration inside the `experimental` block
  experimental: {
    serverActions: {
      bodySizeLimit: '10mb', // Increase the upload limit to 10 Megabytes
    },
  },
};

export default nextConfig;