import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

const categories = [
  { name: "Music", path: "/music" },
  { name: "Movies", path: "/movies" },
   { name: "Gaming", path: "/gaming" },
  { name: "Books", path: "/books" },
  { name: "Fitness & Health", path: "/fitness" },
  { name: "Travel & Tourism", path: "/travel" },
  { name: "Products", path: "/products" },
  { name: "Education", path: "/education" },
  { name: "Food & Drink", path: "/food" },
  { name: "Jobs & Career", path: "/jobs" },
  { name: "Hobbies & Interests", path: "/hobbies" },
  { name: "Sports & Fitness Gear", path: "/sports" }
];

const CategoryGrid = () => {
  const [searchTerm, setSearchTerm] = useState("");
  const navigate = useNavigate();

  const filteredCategories = categories.filter(category =>
    category.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="flex flex-col items-center p-6">
      <input
        type="text"
        placeholder="Search categories..."
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        className="w-3/4 mb-6 h-10 border border-gray-300 rounded-lg p-2 shadow-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-800 dark:text-white dark:border-gray-700"
      />

      <div className="grid grid-cols-4 gap-6 w-3/4">
        {filteredCategories.map((category, index) => (
          <button 
            key={index} 
            onClick={() => navigate(category.path)}
            className="bg-white shadow-md rounded-lg p-8 w-full h-40 flex flex-col items-center justify-center text-lg font-semibold dark:bg-gray-800 dark:text-white text-gray-800 hover:bg-gray-100 transition duration-300"
          >
            <span className="text-4xl">{getEmoji(category.name)}</span>
            <span className="mt-3">{category.name}</span>
          </button>
        ))}
      </div>
    </div>
  );
};

const getEmoji = (category) => {
  const emojiMap = {
    "Music": "🎵", "Movies": "🎬","Gaming": "🎮" , "Books": "📚",
    "Fitness & Health": "💪", "Travel & Tourism": "✈️","Products": "🛍️" , "Education": "🎓",
    "Events": "🎉", "Food & Drink": "🍔", "Fashion": "👗", "News": "📰",
    "Home Automation": "🏠", "Jobs & Career": "💼", "Hobbies & Interests": "🎨", "Sports & Fitness Gear": "⚽"
  };
  return emojiMap[category] || "❓";
};

export default CategoryGrid;
