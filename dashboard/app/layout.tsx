import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Traceboard — PCB Autonomous Lab",
  description: "Observe how a self-improving PCB classifier measures, remembers, experiments, and learns.",
  icons: { icon: "/favicon.svg", shortcut: "/favicon.svg" },
  openGraph: {
    title: "Traceboard — PCB Autonomous Lab",
    description: "Measure, remember, experiment, and learn.",
    images: [{ url: "/og.png", width: 1200, height: 630, alt: "Traceboard PCB autonomous lab" }],
  },
  twitter: { card: "summary_large_image", images: ["/og.png"] },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body className={`${geistSans.variable} ${geistMono.variable}`}>{children}</body></html>;
}
