import type { Metadata } from "next";
import { IBM_Plex_Mono, Inter, Newsreader } from "next/font/google";
import "./globals.css";
import "@uppy/react/css/style.css";
import { APP_NAME, APP_TAGLINE } from "@/lib/branding";

const bodyFont = Inter({
  variable: "--font-body-family",
  subsets: ["latin"],
});

const monoFont = IBM_Plex_Mono({
  variable: "--font-mono-family",
  subsets: ["latin"],
  weight: ["400", "500"],
});

const scholarFont = Newsreader({
  variable: "--font-scholar-family",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  style: ["normal", "italic"],
});

export const metadata: Metadata = {
  title: APP_NAME,
  description: APP_TAGLINE,
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${bodyFont.variable} ${monoFont.variable} ${scholarFont.variable}`}>
      <body className="min-h-screen bg-[#f8fafc] text-slate-900 antialiased">{children}</body>
    </html>
  );
}
