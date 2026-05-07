import type { Metadata } from "next";
import { Bebas_Neue, DM_Sans } from "next/font/google";
import "./globals.css";

const bebas = Bebas_Neue({
  weight: "400",
  subsets: ["latin"],
  variable: "--font-bebas",
});

const dm = DM_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-dm",
});

export const metadata: Metadata = {
  title: "Ai Crate Digger",
  description: "Hunt missing LPs across real shops — smart search for vinyl hunters.",
  icons: {
    icon: [{ url: "/lp.png", type: "image/png" }],
    apple: "/lp.png",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full">
      <body
        className={`${bebas.variable} ${dm.variable} font-sans min-h-full bg-crate-panel text-crate-cream antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
