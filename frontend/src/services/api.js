import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:10000/api';

// Fetch the list of available diseases
export const fetchDiseases = async () => {
  try {
    const response = await axios.get(`${API_URL}/diseases`);
    return response.data;
  } catch (error) {
    console.error('Error fetching diseases:', error);
    throw error;
  }
};

// Fetch the list of available ML algorithms
export const fetchAlgorithms = async () => {
  try {
    const response = await axios.get(`${API_URL}/algorithms`);
    return response.data;
  } catch (error) {
    console.error('Error fetching algorithms:', error);
    throw error;
  }
};

// Train and evaluate models on a single disease dataset
export const trainModels = async (disease, algorithms, options = {}) => {
  try {
    const response = await axios.post(`${API_URL}/train`, {
      disease,
      algorithms,
      test_size: options.testSize || 0.2,
      random_state: options.randomState || 42
    });
    return response.data;
  } catch (error) {
    console.error('Error training models:', error);
    throw error;
  }
};

// Compare models across multiple disease datasets
export const compareModels = async (diseases, algorithms, options = {}) => {
  try {
    const response = await axios.post(`${API_URL}/compare`, {
      diseases,
      algorithms,
      test_size: options.testSize || 0.2,
      random_state: options.randomState || 42
    });
    return response.data;
  } catch (error) {
    console.error('Error comparing models:', error);
    throw error;
  }
}; 
