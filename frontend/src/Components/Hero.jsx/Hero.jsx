import React from 'react';
import { useNavigate } from "react-router-dom";

const Hero = () => {
  const navigate = useNavigate();


  const categories = [
    "Music", "Movies", "Products", "Books", "Fitness & Health",
    "Travel & Tourism", "Gaming", "Education", "Events", "Food",
    "Fashion", "News", "Home Automation", "Jobs & Career",
    "Hobbies & Interests", "Sports & Fitness Gear"
  ];

  return (
    <div className='akhil'>
      <div className='container'>
        {/* Header Section */}
        <div className="flex flex-col items-center py-10">
          <p className="text-blue2 dark:text-purple-200 font-bold text-4xl text-center">
            Infinity Recs: Your Personal Recommendation Assistant
          </p>
          
        </div>

        {/* Search Bar & Dropdown */}
        <div className="mt-6 flex justify-center space-x-4 ">
          {/* <input
            type="text"
            placeholder="Search Categories..."
            className="w-3/4 mb-6 h-10 border border-gray-300 rounded-lg p-2 shadow-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-800 dark:text-white dark:border-gray-700"
          /> */}
          {/* <select className="w-60 h-10 border border-gray-300 rounded-lg p-2 shadow-md focus:ring-2 focus:ring-blue-500 dark:bg-gray-800 dark:text-white dark:border-gray-700">
            <option value="all">All Categories</option>
            {categories.map((category, index) => (
              <option key={index} value={category.toLowerCase().replace(/ & /g, '-').replace(/\s/g, '-')}>
                {category}
              </option>
            ))}
          </select> */}
        </div>
      </div>
    </div>
  );
};

export default Hero;
