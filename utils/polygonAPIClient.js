const axios = require('axios');
const Bottleneck = require('bottleneck');

// Create a Bottleneck limiter for Polygon API
const limiter = new Bottleneck({
    minTime: 13000, // 13 seconds between each request
    maxConcurrent: 1 // Only one request at a time
});

// Base URL for Polygon API
const POLYGON_BASE_URL = 'https://api.polygon.io';

// Wrapper function for Polygon API requests
async function polygonApiRequest(endpoint, params) {
    try {
        const url = `${POLYGON_BASE_URL}${endpoint}`;

        // Use the limiter to schedule the request
        const response = await limiter.schedule(() => axios.get(url, { params }));

        return response.data;
    } catch (error) {
        console.error('Error making Polygon API request:', error.response ? error.response.data : error.message);
        throw error;
    }
}

module.exports = polygonApiRequest;