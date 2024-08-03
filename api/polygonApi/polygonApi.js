const axios = require('axios');
require('dotenv').config();

const apiKey = process.env.POLYGON_API_KEY;

// Log the API key to verify it's being read correctly (remove this in production)
console.log('Polygon API Key:', apiKey);

const getStockQuote = async (symbol) => {
    try {
        const today = new Date().toISOString().split('T')[0]; // Get today's date in YYYY-MM-DD format
        const response = await axios.get(`https://api.polygon.io/v1/open-close/${symbol}/${today}`, {
            params: {
                apiKey: apiKey
            }
        });
        return response.data;
    } catch (error) {
        console.error('Error fetching stock quote:', error.response ? error.response.data : error.message);
        throw new Error('Failed to fetch stock quote');
    }
};

const getHistoricalAggregates = async (symbol, from, to) => {
    try {
        const response = await axios.get(`https://api.polygon.io/v2/aggs/ticker/${symbol}/range/1/day/${from}/${to}`, {
            params: {
                apiKey: apiKey
            }
        });
        return response.data;
    } catch (error) {
        console.error('Error fetching historical aggregates:', error.response ? error.response.data : error.message);
        throw new Error('Failed to fetch historical aggregates');
    }
};

module.exports = {
    getStockQuote,
    getHistoricalAggregates
};