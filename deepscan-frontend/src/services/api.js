import axios from 'axios';

// Use relative roads in production so Vercel rewrites work automatically
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || '';

export async function fetchResults(page = 1, limit = 20) {
  const response = await axios.get(`${API_BASE_URL}/api/results`, {
    params: { page, limit }
  });
  return response.data;
}

export async function analyzeImage(file, description = '') {
  const formData = new FormData();
  formData.append('image', file);
  if (description) formData.append('description', description);

  const response = await axios.post(`${API_BASE_URL}/api/analyze`, formData);
  return response.data;
}

export async function analyzeVideo(file, description = '') {
  const formData = new FormData();
  formData.append('video', file);
  if (description) formData.append('description', description);

  const response = await axios.post(`${API_BASE_URL}/api/analyze-video`, formData);
  return response.data;
}

export async function analyzeMedia(file, description = '') {
  const isVideo = file && file.type.startsWith('video/');
  if (isVideo) {
    return analyzeVideo(file, description);
  }
  return analyzeImage(file, description);
}