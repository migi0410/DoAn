"use client";

import React, { createContext, useContext, useState, useEffect, ReactNode } from "react";

interface ConfigContextType {
  apiUrl: string;
  setApiUrl: (url: string) => void;
}

const ConfigContext = createContext<ConfigContextType | undefined>(undefined);

export const ConfigProvider = ({ children }: { children: ReactNode }) => {
  // Default to env variable or localhost, will be overridden by localStorage in useEffect
  const initialUrl = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");
  const [apiUrl, setApiUrlState] = useState(initialUrl);

  useEffect(() => {
    const savedUrl = localStorage.getItem("API_URL");
    if (savedUrl) {
      setApiUrlState(savedUrl);
    }
  }, []);

  const setApiUrl = (url: string) => {
    // Clean trailing slash
    const cleanUrl = url.endsWith("/") ? url.slice(0, -1) : url;
    setApiUrlState(cleanUrl);
    localStorage.setItem("API_URL", cleanUrl);
  };

  return (
    <ConfigContext.Provider value={{ apiUrl, setApiUrl }}>
      {children}
    </ConfigContext.Provider>
  );
};

export const useConfig = () => {
  const context = useContext(ConfigContext);
  if (context === undefined) {
    throw new Error("useConfig must be used within a ConfigProvider");
  }
  return context;
};
