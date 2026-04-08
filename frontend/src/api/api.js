/**
 * Video Anomaly Detection API Service
 * Production-ready API client with proper error handling
 */

import axios from 'axios';

// Base URL configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 60000, // 60 seconds default
});

// Request interceptor for auth token
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem("token");
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Enhanced error handling
    if (error.code === 'ECONNABORTED') {
      console.error('API Timeout Error:', error.message);
      throw new Error('Request timeout - Backend may be processing or offline');
    } else if (error.code === 'ERR_NETWORK') {
      console.error('Network Error:', error.message);
      throw new Error('Cannot connect to backend - Please ensure backend is running on ' + API_BASE_URL);
    } else if (error.response) {
      console.error('API Error:', error.response.data);
      const detail = error.response.data?.detail;
      const detailMessage = typeof detail === 'string' ? detail : (detail?.message || error.response.data?.message || 'API request failed');
      throw new Error(detailMessage);
    } else {
      console.error('API Error:', error.message);
      throw new Error(error.message || 'Unknown error occurred');
    }
  }
);

// ==================== API Functions (Named Exports) ====================

/**
 * Check system health
 * @returns {Promise<Object>} Health status
 */
export const getHealth = async () => {
  const response = await apiClient.get('/health');
  return response.data;
};

/**
 * Get API information
 * @returns {Promise<Object>} API info
 */
export const getApiInfo = async () => {
  const response = await apiClient.get('/');
  return response.data;
};

/**
 * Get preprocessing proof (sample features)
 * @returns {Promise<Object>} Preprocessing data
 */
export const getPreprocessingProof = async () => {
  const response = await apiClient.get('/preprocessing-proof');
  return response.data;
};

/**
 * Predict anomaly for a random video
 * @returns {Promise<Object>} Prediction result with explanation
 */
export const predictAnomaly = async () => {
  const response = await apiClient.post('/predict');
  return response.data;
};

/**
 * Get memory bank statistics (RA²R)
 * @returns {Promise<Object>} Memory stats
 */
export const getMemoryStats = async () => {
  const response = await apiClient.get('/memory-stats');
  return response.data;
};

/**
 * Get model architecture information
 * @returns {Promise<Object>} Model details
 */
export const getModelInfo = async () => {
  const response = await apiClient.get('/model-info');
  return response.data;
};

/**
 * Upload video for analysis
 * @param {File} videoFile - Video file to upload
 * @param {Function} onProgress - Optional progress callback
 * @returns {Promise} Analysis results
 */
export const uploadVideo = async (videoFile, onProgress) => {
  const formData = new FormData();
  formData.append('video', videoFile);
  
  try {
    const response = await apiClient.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      timeout: 180000, // 3 minutes for video processing
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(percentCompleted);
        }
      },
    });
    
    return response.data;
  } catch (error) {
    // Re-throw with better message
    if (error.message.includes('timeout')) {
      throw new Error('Video analysis is taking longer than expected. Please try a shorter video or try again later.');
    }
    throw error;
  }
};

export const registerUser = async (userData) => {
  const response = await apiClient.post('/register', userData);
  return response.data;
};

export const loginUser = async (credentials) => {
  const response = await apiClient.post('/login', credentials);
  return response.data;
};

export const getUserHistory = async () => {
  const response = await apiClient.get('/user-results');
  return response.data;
};

// Export axios instance for custom requests
export { apiClient };
