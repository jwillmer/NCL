/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        // NCL Brand Colors
        ncl: {
          blue: {
            DEFAULT: "#003A8F",
            dark: "#001F5B",
            light: "#4F83CC",
          },
          gray: {
            DEFAULT: "#6D6E71",
            light: "#D1D3D4",
          },
        },
        // Semantic aliases for components
        primary: {
          DEFAULT: "#003A8F",
          foreground: "#FFFFFF",
        },
        secondary: {
          DEFAULT: "#001F5B",
          foreground: "#FFFFFF",
        },
        accent: {
          DEFAULT: "#4F83CC",
          foreground: "#FFFFFF",
        },
        muted: {
          DEFAULT: "#D1D3D4",
          foreground: "#6D6E71",
        },
        background: "#FFFFFF",
        foreground: "#001F5B",
        card: {
          DEFAULT: "#FFFFFF",
          foreground: "#001F5B",
        },
        border: "#D1D3D4",
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
  plugins: [],
};
