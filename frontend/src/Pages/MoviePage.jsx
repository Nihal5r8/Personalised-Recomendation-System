import React, { useState, useEffect } from 'react';
import { useUser } from '@clerk/clerk-react';
import { FaStar } from 'react-icons/fa';
import { useNavigate, Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';

const MoviePage = () => {
  const { user } = useUser();
  const [movieName, setMovieName] = useState('');
  const [movies, setMovies] = useState([]);
  const [error, setError] = useState('');
  const [userRatings, setUserRatings] = useState({});
  const [watchedStatus, setWatchedStatus] = useState({});
  const [dislikedMovies, setDislikedMovies] = useState({});
  const navigate = useNavigate();

  const fetchRecommendations = async () => {
    try {
      const response = await fetch(`http://127.0.0.1:5003/api/recommend?username=${user.username}&movie=${encodeURIComponent(movieName)}`);
      const data = await response.json();

      if (response.ok) {
        setMovies(data);
        setError('');
      } else {
        setMovies([]);
        setError(data.error);
      }
    } catch (err) {
      console.error('Error fetching recommendations:', err);
      setError('Failed to fetch recommendations.');
    }
  };

  const handleLike = async (tmdbId) => {
    try {
      const response = await fetch('http://127.0.0.1:5003/api/like', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: user.username, tmdb_id: tmdbId }),
      });
      const data = await response.json();

      if (!response.ok) {
        console.error('Error liking movie:', data.error);
      }
    } catch (err) {
      console.error('Error liking movie:', err);
    }
  };

  const handleDislike = async (tmdbId) => {
    try {
      const response = await fetch('http://127.0.0.1:5003/api/dislike', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: user.username, tmdb_id: tmdbId }),
      });
      const data = await response.json();

      if (!response.ok) {
        console.error('Error disliking movie:', data.error);
      } else {
        setDislikedMovies((prev) => ({ ...prev, [tmdbId]: true }));
      }
    } catch (err) {
      console.error('Error disliking movie:', err);
    }
  };

  const handleClick = async (tmdbId) => {
    try {
      await fetch('http://127.0.0.1:5003/api/click', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: user.username, tmdb_id: tmdbId }),
      });
    } catch (err) {
      console.error('Error clicking movie:', err);
    }
  };

  const handleRate = async (tmdbId, rating) => {
    try {
      await fetch('http://127.0.0.1:5003/api/rate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: user.username, tmdb_id: tmdbId, rating }),
      });

      setUserRatings((prev) => ({ ...prev, [tmdbId]: rating }));
    } catch (err) {
      console.error('Error rating movie:', err);
    }
  };

  const handleWatchedChange = async (tmdbId, watched) => {
    try {
      console.log(`Changing watched status for movie ${tmdbId}: ${watched}`);
      setWatchedStatus((prev) => ({ ...prev, [tmdbId]: watched }));
      const response = await fetch('http://127.0.0.1:5003/api/watched', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: user.username, tmdb_id: tmdbId, watched }),
      });
      const data = await response.json();

      if (!response.ok) {
        console.error('Error updating watched status:', data.error);
        setWatchedStatus((prev) => ({ ...prev, [tmdbId]: !watched }));
      } else {
        console.log(`Successfully updated watched status for movie ${tmdbId}: ${watched}`);
      }
    } catch (err) {
      console.error('Error updating watched status:', err);
      setWatchedStatus((prev) => ({ ...prev, [tmdbId]: !watched }));
    }
  };

  useEffect(() => {
    if (user && movies.length) {
      movies.forEach((movie) => {
        fetch(`http://127.0.0.1:5003/api/watched-status?username=${user.username}&tmdb_id=${movie.tmdb_id}`)
          .then((res) => res.json())
          .then((data) => {
            if (data.status !== undefined) {
              setWatchedStatus((prev) => ({ ...prev, [movie.tmdb_id]: data.status }));
            }
          })
          .catch((err) => console.error('Error fetching watched status:', err));
      });
    }
  }, [user, movies]);

  if (!user) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black text-white flex items-center justify-center">
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="text-lg"
        >
          Please log in to view recommendations.
        </motion.p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black text-white flex flex-col items-center justify-start pt-12 pb-12 overflow-x-hidden">
      {/* Back Button with Animation */}
      <motion.div
        initial={{ x: -50, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        <Link
          to="/dash"
          className="flex items-center text-blue-400 hover:text-blue-300 font-semibold mb-6 transition-all duration-300 hover:shadow-[0_0_10px_rgba(59,130,246,0.5)] px-4 py-2 rounded-full bg-gray-800 hover:bg-gray-700"
        >
          ← Back to Dashboard
        </Link>
      </motion.div>

      {/* Header with Animation */}
      <motion.h1
        initial={{ y: -50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6 }}
        className="text-6xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-500 mb-10 text-center"
      >
        🎬 Movie Recommendations
      </motion.h1>

      {/* Recommendation Form with Enhanced Styling */}
      <motion.form
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.5 }}
        onSubmit={(e) => { e.preventDefault(); fetchRecommendations(); }}
        className="mb-12 w-full max-w-xl"
      >
        <div className="flex flex-col items-center space-y-6">
          <input
            type="text"
            placeholder="Enter movie name (e.g., 'Inception')"
            value={movieName}
            onChange={(e) => setMovieName(e.target.value)}
            className="w-full p-5 rounded-xl bg-gray-700 text-white border-2 border-gray-600 focus:border-blue-500 focus:ring-4 ring-blue-500/20 placeholder-gray-400 transition-all duration-300 shadow-md hover:shadow-lg"
          />
          <motion.button
            whileHover={{ scale: 1.05, backgroundColor: "#2563eb" }}
            whileTap={{ scale: 0.95 }}
            type="submit"
            className="w-full px-8 py-4 bg-blue-600 text-white rounded-xl font-bold text-lg transition-all duration-300 shadow-lg hover:shadow-xl"
          >
            Get Recommendations
          </motion.button>
        </div>
      </motion.form>

      {/* Error Message with Animation */}
      <AnimatePresence>
        {error && (
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className="text-red-400 text-lg mb-8 px-4 py-2 bg-red-900/50 rounded-lg shadow-md"
          >
            {error}
          </motion.p>
        )}
      </AnimatePresence>

      {/* Recommendations Display */}
      {movies.length > 0 ? (
        <div className="w-full px-4">
          <motion.h2
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2, duration: 0.5 }}
            className="text-4xl font-semibold text-center mb-10 text-white bg-gradient-to-r from-blue-400 to-purple-300 bg-clip-text"
          >
            Recommended Movies
          </motion.h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-6">
            {movies.map((movie, index) => (
              <motion.div
                key={movie.tmdb_id}
                initial={{ opacity: 0, y: 50 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1, duration: 0.5 }}
                className="bg-gray-800 rounded-2xl overflow-hidden shadow-xl transform transition-all duration-300 hover:scale-105 hover:shadow-2xl hover:bg-gray-700"
              >
                <div className="relative group">
                  {movie.poster_url ? (
                    <a href={movie.tmdb_link} target="_blank" rel="noopener noreferrer">
                      <div className="w-full h-72 bg-black flex items-center justify-center">
                        <img
                          src={movie.poster_url}
                          alt={movie.title}
                          className="w-full h-full object-contain transition-transform duration-300 group-hover:scale-110"
                          onClick={() => handleClick(movie.tmdb_id)}
                        />
                      </div>
                    </a>
                  ) : (
                    <div className="w-full h-72 bg-gray-600 flex items-center justify-center text-gray-400">
                      No Image
                    </div>
                  )}
                  <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-30 transition-opacity duration-300 flex items-center justify-center">
                    <motion.span
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ duration: 0.3 }}
                      className="text-white text-lg font-semibold bg-blue-600/80 px-4 py-2 rounded-full"
                    >
                      View Details
                    </motion.span>
                  </div>
                </div>
                <div className="p-6">
                  <h3 className="text-xl font-bold text-white mb-3 line-clamp-2">{movie.title}</h3>
                  <p className="text-gray-400 text-sm mb-3 line-clamp-2">{movie.genres}</p>
                  {movie.tmdb_link && (
                    <a
                      href={movie.tmdb_link}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="mt-4 inline-block px-5 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-all duration-300 hover:shadow-md"
                      onClick={() => handleClick(movie.tmdb_id)}
                    >
                      More Info
                    </a>
                  )}
                </div>
                {/* Like, Dislike, Rating, Watched Section */}
                <div className="p-4 bg-gray-700 rounded-b-xl border-t border-gray-600">
                  <div className="flex justify-between items-center mb-3">
                    <motion.button
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                      onClick={() => handleLike(movie.tmdb_id)}
                      className="flex items-center px-3 py-1 rounded-full font-semibold transition-all duration-200 bg-gray-600 text-green-400 hover:bg-gray-500"
                    >
                      <span className="mr-1">👍</span> Like
                    </motion.button>
                    <motion.button
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                      onClick={() => handleDislike(movie.tmdb_id)}
                      className={`flex items-center px-3 py-1 rounded-full font-semibold transition-all duration-200 ${
                        dislikedMovies[movie.tmdb_id]
                          ? 'bg-red-500 text-white'
                          : 'bg-gray-600 text-red-400 hover:bg-gray-500'
                      }`}
                    >
                      <span className="mr-1">👎</span> Dislike
                    </motion.button>
                  </div>
                  <div className="flex items-center mb-3">
                    <span className="text-yellow-400 mr-2 font-semibold">Rate:</span>
                    <div className="flex space-x-1">
                      {[1, 2, 3, 4, 5].map((star) => (
                        <motion.button
                          key={star}
                          whileHover={{ scale: 1.2 }}
                          whileTap={{ scale: 0.9 }}
                          onClick={() => handleRate(movie.tmdb_id, star)}
                          className="focus:outline-none"
                        >
                          <FaStar
                            size={24}
                            color={star <= (userRatings[movie.tmdb_id] || 0) ? '#facc15' : '#6b7280'}
                            className="transition-colors duration-200"
                          />
                        </motion.button>
                      ))}
                    </div>
                  </div>
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id={`watched-${movie.tmdb_id}`}
                      checked={watchedStatus[movie.tmdb_id] || false}
                      onChange={(e) => handleWatchedChange(movie.tmdb_id, e.target.checked)}
                      className="h-5 w-5 text-blue-500 rounded focus:ring-blue-500 focus:ring-offset-gray-700 bg-gray-600 border-gray-500"
                    />
                    <label
                      htmlFor={`watched-${movie.tmdb_id}`}
                      className="ml-2 text-gray-300 font-medium"
                    >
                      Watched
                    </label>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      ) : (
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
          className="text-white mt-8 text-lg"
        >
          No recommendations yet. Try entering a movie title!
        </motion.p>
      )}
    </div>
  );
};

export default MoviePage;