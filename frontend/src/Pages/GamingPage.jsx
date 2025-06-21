import React, { useState, useEffect } from 'react';
import { useUser } from "@clerk/clerk-react";
import { Link } from "react-router-dom";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";

const GamingPage = () => {
  const { user, isSignedIn } = useUser();
  const [title, setTitle] = useState("");
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [suggestions, setSuggestions] = useState([]);
  const [userInteractions, setUserInteractions] = useState({});
  const [interactionLoading, setInteractionLoading] = useState({});

  // Fetch suggestions based on query
  const fetchSuggestions = (query) => {
    const mockSuggestions = [
      "Grand Theft Auto V",
      "The Witcher 3: Wild Hunt",
      "Portal 2",
      "Counter-Strike: Global Offensive",
      "Tomb Raider",
      "Elden Ring",
      "Cricket 24"
    ];
    setSuggestions(
      query
        ? mockSuggestions.filter((s) => s.toLowerCase().includes(query.toLowerCase()))
        : []
    );
  };

  // Handle recommendation request
  const handleRecommendationRequest = async (e) => {
    e.preventDefault();
    if (!isSignedIn && !title.trim()) {
      setError("Please sign in or enter a game title to get recommendations.");
      return;
    }

    setLoading(true);
    setError(null);
    setSuggestions([]);
    try {
      const params = isSignedIn && user ? { username: user.username } : { title };
      const response = await axios.get("http://localhost:5002/recommend", { params });
      setRecommendations(response.data);
    } catch (err) {
      const errorMessage =
        err.response?.data?.error || "Failed to fetch recommendations. Please try again.";
      setError(errorMessage);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Handle user interactions
  const handleUserInteraction = async (game, type, value) => {
    const gameId = game.game_id;
    if (interactionLoading[gameId]?.[type]) return; // Prevent duplicate requests

    setInteractionLoading((prev) => ({
      ...prev,
      [gameId]: { ...prev[gameId], [type]: true },
    }));

    const prev = userInteractions[gameId] || {};
    const updated = { ...prev, [type]: value };

    const updatedInteractions = {
      ...userInteractions,
      [gameId]: updated,
    };
    setUserInteractions(updatedInteractions);

    try {
      await axios.post("http://localhost:5002/interact", {
        username: user.username,
        game: {
          title: game.title,
          genres: game.genres,
          platforms: game.platforms,
          rating: game.rating,
          released: game.released,
          image_url: game.image_url,
          link: game.link,
        },
        interaction: {
          rating: updated.rating || 0,
          liked: updated.liked || false,
          watched: updated.watched || false,
          clicked: updated.clicked || false,
        },
      });
    } catch (err) {
      console.error("Error saving interaction:", err);
      setError("Failed to save interaction. Please try again.");
    } finally {
      setInteractionLoading((prev) => ({
        ...prev,
        [gameId]: { ...prev[gameId], [type]: false },
      }));
    }
  };

  // Handle game click
  const handleGameClick = (game) => {
    window.open(game.link, "_blank");
    handleUserInteraction(game, "clicked", true);
  };

  // Clear error after 5 seconds
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(null), 5002);
      return () => clearTimeout(timer);
    }
  }, [error]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-800 text-gray-100 flex flex-col items-center justify-start pt-20 pb-20 overflow-x-hidden relative">
      {/* Background effects */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,#3b82f6_0%,transparent_70%)] opacity-10 pointer-events-none"></div>
      <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/stardust.png')] opacity-5 pointer-events-none"></div>

      {/* Back button */}
      <motion.div
        initial={{ x: -80, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
        className="z-20"
      >
        <Link
          to="/dash"
          className="flex items-center gap-2 text-blue-300 hover:text-blue-200 font-medium mb-10 transition-all duration-300 px-6 py-3 rounded-xl bg-gray-800/40 backdrop-blur-md border border-gray-700/50 hover:bg-gray-700/50 hover:shadow-[0_0_20px_rgba(59,130,246,0.4)]"
        >
          <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
            <path d="M15.41 7.41L14 6l-6 6 6 6 1.41-1.41L10.83 12z" />
          </svg>
          Back to Dashboard
        </Link>
      </motion.div>

      {/* Title */}
      <motion.h1
        initial={{ y: -80, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.9, ease: "easeOut" }}
        className="z-20 text-8xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-purple-500 to-pink-500 mb-16 text-center drop-shadow-2xl tracking-tight"
      >
        🎮 Discover Your Next Game
      </motion.h1>

      {/* Search */}
      <motion.form
        initial={{ scale: 0.85, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.7, ease: "easeOut" }}
        onSubmit={handleRecommendationRequest}
        className="z-20 mb-16 w-full max-w-3xl relative"
        aria-label="Game recommendation search form"
      >
        <div className="relative">
          <input
            type="text"
            value={title}
            onChange={(e) => {
              setTitle(e.target.value);
              fetchSuggestions(e.target.value);
            }}
            placeholder="Search for a game (e.g., 'Elden Ring')"
            className="w-full p-6 rounded-xl bg-gray-800/50 text-white border-2 border-gray-700/50 focus:border-blue-500 focus:ring-4 ring-blue-500/40 placeholder-gray-500 transition-all duration-300 shadow-xl hover:shadow-2xl outline-none backdrop-blur-md"
            aria-label="Search for a game title"
            disabled={loading}
          />
          <AnimatePresence>
            {suggestions.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.3 }}
                className="absolute w-full mt-2 bg-gray-800/90 rounded-xl shadow-2xl border border-gray-700/50 backdrop-blur-md max-h-60 overflow-y-auto"
                role="listbox"
                aria-label="Game title suggestions"
              >
                {suggestions.map((suggestion, index) => (
                  <button
                    key={index}
                    type="button"
                    onClick={() => {
                      setTitle(suggestion);
                      setSuggestions([]);
                    }}
                    className="w-full text-left px-6 py-3 text-gray-200 hover:bg-gray-700/70 transition-all duration-200 border-b border-gray-700/30 last:border-b-0"
                    role="option"
                    aria-selected={title === suggestion}
                  >
                    {suggestion}
                  </button>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
        <motion.button
          whileHover={{ scale: 1.08, backgroundColor: "#3b82f6" }}
          whileTap={{ scale: 0.95 }}
          disabled={loading}
          type="submit"
          className="w-full mt-6 px-12 py-5 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl font-bold text-xl transition-all duration-300 disabled:bg-gray-600 disabled:cursor-not-allowed shadow-2xl hover:shadow-[0_0_25px_rgba(59,130,246,0.5)]"
          aria-label="Get game recommendations"
          aria-busy={loading}
        >
          {loading ? (
            <span className="flex items-center justify-center gap-3">
              <svg className="animate-spin h-7 w-7 text-white" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Loading...
            </span>
          ) : (
            "Discover Now"
          )}
        </motion.button>
      </motion.form>

      {/* Error */}
      <AnimatePresence>
        {error && (
          <motion.p
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -40 }}
            transition={{ duration: 0.5 }}
            className="z-20 text-red-300 text-xl mb-12 px-8 py-4 bg-gray-800/50 rounded-xl shadow-inner border border-gray-700/50 backdrop-blur-md"
            role="alert"
          >
            {error}
          </motion.p>
        )}
      </AnimatePresence>

      {/* Recommendations */}
      <div className="z-20 w-full max-w-[90rem] px-10">
        {loading ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-12">
            {[...Array(5)].map((_, i) => (
              <div
                key={i}
                className="bg-gray-800/70 rounded-xl h-96 animate-pulse border border-gray-700/50"
              ></div>
            ))}
          </div>
        ) : recommendations.length > 0 ? (
          <>
            <motion.h2
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4, duration: 0.7 }}
              className="text-6xl font-bold text-center mb-16 text-transparent bg-gradient-to-r from-blue-400 via-purple-500 to-pink-500 bg-clip-text drop-shadow-2xl"
            >
              Your Game Recommendations
            </motion.h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-12">
              {recommendations.map((game) => {
                const interaction = userInteractions[game.game_id] || {};
                const { rating = 0, liked = null, watched = false } = interaction;
                const isInteractionLoading = interactionLoading[game.game_id] || {};

                return (
                  <motion.div
                    key={game.game_id}
                    initial={{ opacity: 0, y: 80 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    className="bg-gray-900/90 rounded-2xl overflow-hidden shadow-2xl hover:shadow-[0_0_30px_rgba(59,130,246,0.3)]"
                  >
                    <div className="relative group">
                      {game.image_url ? (
                        <img
                          src={game.image_url}
                          alt={game.title}
                          className="w-full h-96 object-cover cursor-pointer"
                          onClick={() => handleGameClick(game)}
                          loading="lazy"
                        />
                      ) : (
                        <div
                          className="w-full h

-96 bg-gray-800 flex items-center justify-center text-gray-600 cursor-pointer"
                          onClick={() => handleGameClick(game)}
                        >
                          No Image Available
                        </div>
                      )}
                    </div>
                    <div className="p-6 space-y-3">
                      <h3 className="text-2xl font-bold text-white">{game.title}</h3>
                      <p className="text-gray-300 text-sm">Genres: {game.genres}</p>
                      <p className="text-gray-300 text-sm">Platforms: {game.platforms}</p>
                      <p className="text-yellow-400 text-sm">Rating: {game.rating}</p>

                      {/* Star Rating */}
                      <div className="flex items-center gap-1 mt-2" role="group" aria-label={`Rate ${game.title}`}>
                        {[1, 2, 3, 4, 5].map((star) => (
                          <button
                            key={star}
                            type="button"
                            onClick={() => handleUserInteraction(game, "rating", star)}
                            disabled={isInteractionLoading.rating}
                            className={`w-6 h-6 cursor-pointer focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                              rating >= star ? 'text-yellow-400' : 'text-gray-500'
                            } ${isInteractionLoading.rating ? 'opacity-50 cursor-not-allowed' : 'hover:text-yellow-300'}`}
                            aria-label={`Rate ${star} stars`}
                          >
                            <svg fill="currentColor" viewBox="0 0 24 24">
                              <path d="M12 17.75l-6.61 3.47 1.26-7.46-5.43-5.29 7.49-1.09L12 0l3.29 6.67 7.49 1.09-5.43 5.29 1.26 7.46z" />
                            </svg>
                          </button>
                        ))}
                        <span className="ml-1 text-sm text-gray-400">({rating}/5)</span>
                      </div>

                      {/* Like/Dislike Buttons */}
                      <div className="flex gap-3 mt-2">
                        <button
                          onClick={() => handleUserInteraction(game, "liked", liked === true ? null : true)}
                          disabled={isInteractionLoading.liked}
                          className={`text-lg focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                            liked === true ? 'text-blue-500' : 'text-gray-400'
                          } ${isInteractionLoading.liked ? 'opacity-50 cursor-not-allowed' : 'hover:text-blue-300'} transition-all`}
                          aria-label={liked === true ? 'Unlike' : 'Like'}
                          aria-pressed={liked === true}
                        >
                          👍 {isInteractionLoading.liked && liked === true ? '...' : 'Like'}
                        </button>
                        <button
                          onClick={() => handleUserInteraction(game, "liked", liked === false ? null : false)}
                          disabled={isInteractionLoading.liked}
                          className={`text-lg focus:outline-none focus:ring-2 focus:ring-red-500 ${
                            liked === false ? 'text-red-500' : 'text-gray-400'
                          } ${isInteractionLoading.liked ? 'opacity-50 cursor-not-allowed' : 'hover:text-red-300'} transition-all`}
                          aria-label={liked === false ? 'Remove dislike' : 'Dislike'}
                          aria-pressed={liked === false}
                        >
                          👎 {isInteractionLoading.liked && liked === false ? '...' : 'Dislike'}
                        </button>
                      </div>

                      {/* Watched Checkbox */}
                      <div className="mt-2">
                        <label
                          htmlFor={`watched-${game.game_id}`}
                          className="inline-flex items-center space-x-2 cursor-pointer"
                        >
                          <input
                            type="checkbox"
                            id={`watched-${game.game_id}`}
                            checked={watched}
                            onChange={() => handleUserInteraction(game, "watched", !watched)}
                            disabled={isInteractionLoading.watched}
                            className="w-5 h-5 rounded-xl text-blue-600 focus:ring-blue-500 disabled:opacity-50"
                            aria-label={`Mark ${game.title} as watched`}
                          />
                          <span className="text-sm">
                            {isInteractionLoading.watched ? 'Updating...' : 'Watched'}
                          </span>
                        </label>
                      </div>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </>
        ) : (
          <p className="col-span-full text-center text-xl text-gray-400">
            {recommendations.length === 0 && !loading
              ? "No recommendations found. Try searching for a game or sign in for personalized suggestions!"
              : ""}
          </p>
        )}
      </div>
    </div>
  );
};

export default GamingPage;