import React from 'react';

import AuthContextProvider from './AuthContext';

const ContextProvider = ({children}) => {
  return (
    <AuthContextProvider>
    {children}
    </AuthContextProvider>
  );
};

export default ContextProvider;