import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Switchback AI — Every Experiment Makes the Next One Smarter",
  description: "A self-improving experiment agent that turns every model run into a head start on the next problem.",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
