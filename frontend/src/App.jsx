import React from "react";
import { createBrowserRouter,createRoutesFromElements,RouterProvider,Route, BrowserRouter, Routes } from "react-router-dom";
import { ClerkProvider } from "@clerk/clerk-react";
import Dash from "./Dash";
import Home from "./Home";
import MusicPage from "./Pages/MusicPage";
import MoviePage from "./Pages/MoviePage";
import ProductsPage from "./Pages/ProductsPage";
import BooksPage from "./Pages/BooksPage";
import FitnessPage from "./Pages/FitnessPage";
import TravelPage from "./Pages/TravelPage";
import GamingPage from "./Pages/GamingPage";
import EducationPage from "./Pages/EducationPage";
import FoodPage from "./Pages/FoodPage";
import JobsPage from "./Pages/JobsPage";
import HobbiesPage from "./Pages/HobbiesPage";
import SportsPage from "./Pages/SportsPage";
// hii

const App = () => {
  return (
    <>
    <BrowserRouter>
    <Routes>
    <Route index element = {<Home/>}/>
    <Route path="/dash" element={<Dash/>}/>
    <Route path="/music" element={<MusicPage />} />
    <Route path="/movies" element={<MoviePage />} />
    <Route path="/products" element={<ProductsPage />} />
    <Route path="/books" element={<BooksPage />} />
    <Route path="/fitness" element={<FitnessPage />} />
    <Route path="/travel" element={<TravelPage />} />
    <Route path="/gaming" element={<GamingPage />} />
    <Route path="/education" element={<EducationPage />} />
    <Route path="/food" element={<FoodPage />} />
    <Route path="/jobs" element={<JobsPage />} />
    <Route path="/hobbies" element={<HobbiesPage />} />
    <Route path="/sports" element={<SportsPage />} />
    
    </Routes>
    </BrowserRouter>
    </>
  );
};

export default App;
