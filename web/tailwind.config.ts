import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // Maran Tankers Brand Colors
        ncl: {
          blue: {
            DEFAULT: "#1B365D",
            dark: "#0F2340",
            light: "#3D5A80",
          },
          gray: {
            DEFAULT: "#5C6670",
            light: "#C9CED3",
          },
        },
        // Semantic aliases for components
        primary: {
          DEFAULT: "#1B365D",
          foreground: "#FFFFFF",
        },
        secondary: {
          DEFAULT: "#0F2340",
          foreground: "#FFFFFF",
        },
        accent: {
          DEFAULT: "#3D5A80",
          foreground: "#FFFFFF",
        },
        muted: {
          DEFAULT: "#C9CED3",
          foreground: "#5C6670",
        },
        background: "#FFFFFF",
        foreground: "#0F2340",
        card: {
          DEFAULT: "#FFFFFF",
          foreground: "#0F2340",
        },
        border: "#C9CED3",
        destructive: {
          DEFAULT: "#DC2626",
          foreground: "#FFFFFF",
        },
      },
      borderRadius: {
        lg: "0.5rem",
        md: "0.375rem",
        sm: "0.25rem",
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
};

export default config;
