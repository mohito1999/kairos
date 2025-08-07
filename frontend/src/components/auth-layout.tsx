import Image from 'next/image';
import React from 'react';

interface AuthLayoutProps {
  children: React.ReactNode;
  imageUrl: string;
  imageAlt: string;
}

export default function AuthLayout({ children, imageUrl, imageAlt }: AuthLayoutProps) {
  return (
    // CRITICAL FIX: Added `h-screen` and `overflow-hidden` to the parent container.
    // This forces the entire component to be exactly the height of the screen and hides any scrollbars.
    <div className="w-full lg:grid lg:h-screen lg:grid-cols-2 overflow-hidden">
      {/* Left Column: Image */}
      <div className="hidden bg-muted lg:block h-full">
        <Image
          src={imageUrl}
          alt={imageAlt}
          width="1920"
          height="1080"
          className="h-full w-full object-cover object-top brightness-75" // Removed dark mode filters for now to match your image
          priority
        />
      </div>
      {/* Right Column: Form */}
      <div className="flex items-center justify-center h-screen overflow-y-auto py-12">
        {/* CRITICAL FIX: Added `h-screen` and `overflow-y-auto`.
            This makes the form column itself scrollable on small screens if needed,
            without creating a scrollbar on the whole page. */}
        <div className="mx-auto grid w-[350px] gap-6">
          {children}
        </div>
      </div>
    </div>
  );
}