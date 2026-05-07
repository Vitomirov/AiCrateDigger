import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        crate: {
          night: "#0f0d0b",
          panel: "#1c1410",
          rust: "#3d241a",
          cream: "#f5e6d3",
          amber: "#e8b86d",
          gold: "#c9a227",
          grove: "#0a0908",
        },
      },
      fontFamily: {
        slab: ["var(--font-bebas)", "Impact", "sans-serif"],
        sans: ["var(--font-dm)", "system-ui", "sans-serif"],
      },
      boxShadow: {
        platter: "0 18px 50px rgba(0, 0, 0, 0.6)",
      },
      dropShadow: {
        platter: "0 20px 40px rgba(0,0,0,0.55)",
      },
    },
  },
  plugins: [],
};

export default config;
