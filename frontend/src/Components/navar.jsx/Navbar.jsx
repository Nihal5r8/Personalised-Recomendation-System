import React from 'react';
import DarkMode from './DarkMode';


const NavLinks = [
    {
        id: 1,
        name: 'Dashboard',
        link: '/#',
    },
    {
        id: 2,
        name: 'History',
        link: '/#',
    },
    {
        id: 3,
        name: 'Support',
        link: '/#',
    },
];

const Navbar = () => {
  return (
    <>
      <div className='bg-white shadow-md fixed top-0 left-0 w-full z-50 dark:bg-gray-800'>
        <div className='container flex justify-between py-4 sm:py-3 px-6'>
          {/* Logo */}
          <div className='font-bold text-blue-600 text-2xl'>
            <a href="/#">Infinity Recs</a>
          </div>
          
          {/* Navigation Links */}
          <div className="flex items-center">
            <ul className="flex items-center gap-10">
              {NavLinks.map(({ id, name, link }) => (
                <li key={id}>
                  <a href={link} className="hover:text-blue-500 dark:text-white">{name}</a>
                </li>
              ))}
            </ul>
            
            {/* Login Button */}
            <div className='px-3 py-1'>
              <DarkMode/>
            </div>
            
          </div>
        </div>
      </div>

      
      <div className="h-16"></div>
    </>
  );
};

export default Navbar;
