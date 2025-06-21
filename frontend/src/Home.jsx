import React, { useEffect } from "react";
import { SignInButton, UserButton, useUser } from "@clerk/clerk-react";
import { Link } from "react-router-dom"; // Add Link for navigation
import axios from "axios";

const Home = () => {
  const { user, isSignedIn } = useUser();

  useEffect(() => {
    const saveUserData = async () => {
      if (isSignedIn && user) {
        try {
          const userData = {
            user_id: user.id,
            email: user.primaryEmailAddress.emailAddress,
            username: user.username || null,
          };
          await axios.post("http://localhost:5000/user", userData);
          console.log("User data saved:", userData);

          const response = await axios.get(`http://localhost:5000/user/${user.id}`);
          console.log("Fetched user data:", response.data);
        } catch (error) {
          console.error("Error saving user data:", error);
        }
      }
    };
    saveUserData();
  }, [isSignedIn, user]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-white">
      {/* Navbar */}
      <nav className="flex justify-between items-center p-5 border-b border-gray-700">
        <div className="text-2xl font-bold text-blue-400">Infinity Recs</div>
        <UserButton />
      </nav>

      {/* Main Content */}
      <div className="flex flex-col items-center justify-center text-center py-20 px-4">
        <h1 className="text-5xl font-bold mb-6">
          Welcome to <span className="text-blue-400">InfinityRecs</span>
        </h1>
        <p className="text-xl text-gray-300 mb-8 max-w-2xl">
          Discover personalized recommendations for movies, music, books, and more. 
          InfinityRecs uses advanced algorithms to curate the perfect list just for you.
        </p>

        {!isSignedIn ? (
          <SignInButton forceRedirectUrl="/dash" asChild>
            <button className="mt-6 px-6 py-3 bg-blue-500 text-center hover:bg-blue-600 text-white rounded-lg font-semibold">
              Sign In to Continue
            </button>
          </SignInButton>
        ) : (
          <Link to="/dash">
            <button className="mt-6 px-6 py-3 bg-blue-500 text-center hover:bg-blue-600 text-white rounded-lg font-semibold">
              Go to Dashboard
            </button>
          </Link>
        )}
      </div>

      {/* Why Choose Section */}
      <div className="bg-gray-800 py-16">
        <div className="max-w-6xl mx-auto px-4">
          <h2 className="text-3xl font-bold text-center mb-12">Why Choose InfinityRecs?</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="bg-gray-700 p-6 rounded-lg shadow-lg">
              <h3 className="text-xl font-semibold mb-4">Personalized Recommendations</h3>
              <p className="text-gray-300">
                Get recommendations tailored to your unique tastes and preferences.
              </p>
            </div>
            <div className="bg-gray-700 p-6 rounded-lg shadow-lg">
              <h3 className="text-xl font-semibold mb-4">Wide Range of Categories</h3>
              <p className="text-gray-300">
                Explore recommendations for movies, music, books, games, and more.
              </p>
            </div>
            <div className="bg-gray-700 p-6 rounded-lg shadow-lg">
              <h3 className="text-xl font-semibold mb-4">Easy to Use</h3>
              <p className="text-gray-300">
                A simple and intuitive interface designed for seamless navigation.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="text-center py-6 border-t border-gray-700">
        <p className="text-gray-400">
          © {new Date().getFullYear()} InfinityRecs. All rights reserved.
        </p>
      </footer>
    </div>
  );
};

export default Home;