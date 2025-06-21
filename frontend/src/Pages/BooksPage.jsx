import React, { useState, useEffect, useCallback } from 'react';
import { useUser } from "@clerk/clerk-react";
import { Link } from "react-router-dom";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";
import debounce from "lodash.debounce";

const BooksPage = () => {
  const { user, isSignedIn } = useUser();
  const [title, setTitle] = useState("");
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [suggestions, setSuggestions] = useState([]);
  const [userInteractions, setUserInteractions] = useState({});

  // Fetch suggestions from backend
  const fetchSuggestions = async (query) => {
    if (!query) {
      setSuggestions([]);
      return;
    }
    try {
      const response = await axios.get("http://localhost:5001/suggest", { params: { query } });
      setSuggestions(response.data);
    } catch (err) {
      console.error("Error fetching suggestions:", err);
      setSuggestions([]);
    }
  };

  const debouncedFetchSuggestions = useCallback(debounce(fetchSuggestions, 300), []);

  // Fetch user-based recommendations
  const fetchUserRecommendations = async () => {
    if (!isSignedIn || !user) {
      setError("Please sign in to get personalized recommendations.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get("http://localhost:5001/recommend", { params: { username: user.username } });
      setRecommendations(response.data);
      const initialInteractions = response.data.reduce((acc, book) => ({
        ...acc,
        [book.book_id]: { rating: 0, liked: false, disliked: false, read: false, clicked: false }
      }), {});
      setUserInteractions(initialInteractions);
    } catch (err) {
      setError(err.response?.data?.error || "Failed to fetch recommendations. Please try again.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Handle recommendation request (title-based or user-based)
  const handleRecommendationRequest = async (e) => {
    e.preventDefault();
    if (!isSignedIn || !user) {
      setError("Please sign in to get recommendations.");
      return;
    }
    if (title.trim()) {
      setLoading(true);
      setError(null);
      setSuggestions([]);
      try {
        const response = await axios.get("http://localhost:5001/recommend", { params: { title } });
        setRecommendations(response.data);
        const initialInteractions = response.data.reduce((acc, book) => ({
          ...acc,
          [book.book_id]: { rating: 0, liked: false, disliked: false, read: false, clicked: false }
        }), {});
        setUserInteractions(initialInteractions);
      } catch (err) {
        setError(err.response?.data?.error || "Failed to fetch recommendations. Please try again.");
        console.error(err);
      } finally {
        setLoading(false);
      }
    } else {
      fetchUserRecommendations();
    }
  };

  // Handle user interactions (rating, liked, disliked, read, clicked)
  const handleUserInteraction = async (e, book, type, value) => {
    e.preventDefault();
    const prev = userInteractions[book.book_id] || {};
    let updated = { ...prev, [type]: value };

    // Ensure liked and disliked are mutually exclusive
    if (type === "liked" && value) {
      updated = { ...updated, liked: true, disliked: false };
    } else if (type === "disliked" && value) {
      updated = { ...updated, liked: false, disliked: true };
    } else if (type === "liked" && !value) {
      updated = { ...updated, liked: false };
    } else if (type === "disliked" && !value) {
      updated = { ...updated, disliked: false };
    }

    const updatedInteractions = {
      ...userInteractions,
      [book.book_id]: updated,
    };
    setUserInteractions(updatedInteractions);

    try {
      await axios.post("http://localhost:5001/save_interaction", {
        username: user.username,
        title: book.title,
        rating: updated.rating || null,
        liked: updated.disliked ? false : updated.liked || false,
        read: updated.read || false,
        clicked: updated.clicked || false
      });
    } catch (err) {
      console.error("Error saving interaction:", err.response ? err.response.data : err);
    }
  };

  // Handle poster click (opens link and records interaction)
  const handlePosterClick = (e, book) => {
    e.preventDefault();
    window.open(book.link, "_blank");
    handleUserInteraction(e, book, "clicked", true);
  };

  // Fetch user-based recommendations on mount
  useEffect(() => {
    if (isSignedIn && user && !title) {
      fetchUserRecommendations();
    }
  }, [isSignedIn, user]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-800 text-gray-100 flex flex-col items-center justify-start pt-20 pb-20 overflow-x-hidden relative">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,#3b82f6_0%,transparent_70%)] opacity-10 pointer-events-none" />
      <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/stardust.png')] opacity-5 pointer-events-none" />

      <motion.div initial={{ x: -80, opacity: 0 }} animate={{ x: 0, opacity: 1 }} transition={{ duration: 0.8 }} className="z-20">
        <Link to="/dash" className="flex items-center gap-2 text-blue-300 hover:text-blue-200 font-medium mb-10 px-6 py-3 rounded-xl bg-gray-800/40 border border-gray-700/50 hover:bg-gray-700/50">
          <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M15.41 7.41L14 6l-6 6 6 6 1.41-1.41L10.83 12z"/></svg>
          Back to Dashboard
        </Link>
      </motion.div>

      <motion.h1 initial={{ y: -80, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ duration: 0.9 }} className="z-20 text-8xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-purple-500 to-pink-500 mb-16 text-center">
        📚 Discover Your Next Book
      </motion.h1>

      <motion.form onSubmit={handleRecommendationRequest} initial={{ scale: 0.85, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ duration: 0.7 }} className="z-20 mb-16 w-full max-w-3xl relative">
        <div className="relative">
          <input
            value={title}
            onChange={(e) => {
              setTitle(e.target.value);
              debouncedFetchSuggestions(e.target.value);
            }}
            placeholder="Search for a book or discover personalized recommendations..."
            className="w-full p-6 rounded-xl bg-gray-800/50 border-2 border-gray-700/50 focus:border-blue-500 ring-blue-500/40 placeholder-gray-500 shadow-xl hover:shadow-2xl backdrop-blur-md"
          />
          <AnimatePresence>
            {suggestions.length > 0 && (
              <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} transition={{ duration: 0.3 }}
                className="absolute w-full mt-2 bg-gray-800/90 rounded-xl shadow-2xl border border-gray-700/50 backdrop-blur-md max-h-60 overflow-y-auto">
                {suggestions.map((s, i) => (
                  <button key={i} type="button" onClick={() => { setTitle(s); setSuggestions([]); }}
                    className="w-full text-left px-6 py-3 text-gray-200 hover:bg-gray-700/70 border-b border-gray-700/30 last:border-b-0">
                    {s}
                  </button>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <motion.button type="submit" disabled={loading} whileHover={{ scale: 1.08 }} whileTap={{ scale: 0.95 }}
          className="w-full mt-6 px-12 py-5 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl font-bold text-xl shadow-2xl hover:shadow-[0_0_25px_rgba(59,130,246,0.5)] focus:outline-none focus:ring-4">
          {loading ? (
            <span className="flex items-center justify-center gap-3">
              <svg className="animate-spin h-7 w-7" viewBox="0 0 24 24">
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" className="opacity-25" />
                <path fill="currentColor" className="opacity-75" d="M4 12a8 8 0 018-8V0C5.37 0 0 5.37 0 12h4z" />
              </svg> Loading...
            </span>
          ) : "Discover Now"}
        </motion.button>
      </motion.form>

      <AnimatePresence>
        {error && (
          <motion.p initial={{ opacity: 0, y: 40 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -40 }}
            className="z-20 text-red-300 text-xl mb-12 px-8 py-4 bg-gray-800/50 rounded-xl border border-gray-700/50">
            {error}
          </motion.p>
        )}
      </AnimatePresence>

      <div className="z-20 w-full max-w-[90rem] px-10">
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-12">
          {loading ? (
            [...Array(5)].map((_, i) => <div key={i} className="bg-gray-800/70 rounded-xl h-48 animate-pulse border" />)
          ) : recommendations.length > 0 ? (
            recommendations.map((book) => {
              const interaction = userInteractions[book.book_id] || {};
              const { rating = 0, liked = false, disliked = false, read = false } = interaction;
              return (
                <motion.div key={book.book_id} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} transition={{ duration: 0.5 }}
                  className="bg-gray-800/60 rounded-xl p-5 relative group">
                  {book.img ? (
                    <img
                      src={book.img}
                      alt={book.title}
                      className="w-full h-60 object-cover rounded-lg mb-4 cursor-pointer"
                      onClick={(e) => handlePosterClick(e, book)}
                    />
                  ) : (
                    <div className="w-full h-60 bg-gray-600 flex items-center justify-center text-gray-400 rounded-lg mb-4">
                      No Image
                    </div>
                  )}
                  <h3 className="text-xl font-semibold mb-2">{book.title}</h3>
                  <p className="text-sm text-gray-400">{book.author}</p>
                  <p className="text-sm text-gray-500">{book.genre}</p>
                  <p className="text-sm text-gray-500">Rating: {book.rating.toFixed(2)}</p>
                  <p className="text-sm text-gray-500">Total Ratings: {book.totalratings}</p>

                  <div className="flex items-center gap-1 mt-4">
                    {[1, 2, 3, 4, 5].map(star => (
                      <svg key={star} className={`w-6 h-6 cursor-pointer ${rating >= star ? 'text-yellow-400' : 'text-gray-500'}`}
                        onClick={(e) => handleUserInteraction(e, book, "rating", star)} fill="currentColor" viewBox="0 0 24 24">
                        <path d="M12 17.75l-6.61 3.47 1.26-7.46-5.43-5.29 7.49-1.09L12 0l3.29 6.67 7.49 1.09-5.43 5.29 1.26 7.46z" />
                      </svg>
                    ))}
                  </div>

                  <div className="flex gap-3 mt-4">
                    <button type="button" onClick={(e) => handleUserInteraction(e, book, "liked", !liked)}
                      className={`text-xl ${liked ? 'text-blue-500' : 'text-gray-400'}`}>👍</button>
                    <button type="button" onClick={(e) => handleUserInteraction(e, book, "disliked", !disliked)}
                      className={`text-xl ${disliked ? 'text-red-500' : 'text-gray-400'}`}>👎</button>
                  </div>

                  <div className="flex items-center mt-4">
                    <input type="checkbox" checked={read} onChange={(e) => handleUserInteraction(e, book, "read", e.target.checked)}
                      className="cursor-pointer" />
                    <span className="ml-2 text-sm" title={read ? "Read" : ""}>Read</span>
                  </div>
                </motion.div>
              );
            })
          ) : (
            <p className="col-span-full text-center text-xl text-gray-400">No recommendations found. Try searching or discover personalized recommendations.</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default BooksPage;