/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: "class", // Enable dark mode using a class
  content: ["./src/**/*.{js,jsx,ts,tsx}"],

  theme: {
    extend: {
      colors: {
        blue1: '#2563eb',
        blue2: '#1e40af',
        darkyy: '#1c3652',
      
      },
      container: {
        center: true,
        padding: {
          Deafult: '1rem',
          sm: '2rem',
          lg: '3rem',
          xl: '4rem',
        },
      } 
    },
  },
  plugins: [],
}

