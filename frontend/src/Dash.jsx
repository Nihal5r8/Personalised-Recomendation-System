import React, { useEffect, useState } from "react";
import Navbar from "./Components/navar.jsx/Navbar";
import Hero from "./Components/Hero.jsx/Hero";
import CategoryGrid from "./Components/CategoryGrid.jsx/CategoryGrid";
import { Routes, Route } from "react-router-dom";
import { useUser } from "@clerk/clerk-react";
import axios from "axios";

const Dash = () => {
  const { user, isSignedIn } = useUser();
  const [userData, setUserData] = useState(null);

  useEffect(() => {
    const fetchUserData = async () => {
      if (isSignedIn && user) {
        try {
          const response = await axios.get(`http://localhost:5000/user/${user.id}`);
          setUserData(response.data);
        } catch (error) {
          console.error("Error fetching user data:", error);
        }
      }
    };
    fetchUserData();
  }, [isSignedIn, user]);

  return (
    <div className="overflow-x-hidden min-h-screen bg-gradient-to-b from-blue-100 to-purple-200 dark:from-darkyy dark:to-purple-300">
      <Navbar />
      {userData && (
        <div className="p-4 text-center">
          <p className="text-lg">Welcome, {userData.email}!</p>
        </div>
      )}
      <Routes>
        <Route path="/" element={<Hero />} />
      </Routes>
      <CategoryGrid />
    </div>
  );
};

export default Dash;