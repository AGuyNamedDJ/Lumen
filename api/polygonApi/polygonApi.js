const axios = require('axios');
require('dotenv').config();

const apiKey = process.env.POLYGON_API_KEY;
const BASE_URL = 'https://api.polygon.io';

const getStockQuote = async (symbol) => {
    try {
        const today = new Date().toISOString().split('T')[0]; // Get today's date in YYYY-MM-DD format
        const response = await axios.get(`${BASE_URL}/v1/open-close/${symbol}/${today}`, {
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

const getTechnicalIndicator = async (symbol, indicator, timespan, window, seriesType) => {
    console.log('Fetching technical indicator with params:', { symbol, indicator, timespan, window, seriesType });
    try {
        const response = await axios.get(`${BASE_URL}/v1/indicators/${indicator}/${symbol}`, {
            params: {
                timespan: timespan,
                window: window,
                series_type: seriesType,
                apiKey: apiKey
            }
        });
        console.log('Technical indicator response:', response.data);
        return response.data;
    } catch (error) {
        console.error('Error fetching technical indicator:', error.response ? error.response.data : error.message);
        throw new Error('Failed to fetch technical indicator');
    }
};

const getHistoricalAggregates = async (symbol, from, to, timespan) => {
    console.log('Fetching historical aggregates with params:', { symbol, from, to, timespan });
    try {
        const response = await axios.get(`${BASE_URL}/v2/aggs/ticker/${symbol}/range/1/${timespan}/${from}/${to}`, {
            params: {
                apiKey: apiKey
            }
        });
        console.log('Historical aggregates response:', response.data);
        return response.data;
    } catch (error) {
        console.error('Error fetching historical aggregates:', error.response ? error.response.data : error.message);
        throw new Error('Failed to fetch historical aggregates');
    }
};

const searchSymbols = async (query) => {
    try {
        const response = await axios.get(`https://api.polygon.io/v3/reference/tickers`, {
            params: {
                search: query,
                apiKey: apiKey
            }
        });
        return response.data;
    } catch (error) {
        console.error('Error searching symbols:', error.response ? error.response.data : error.message);
        throw new Error('Failed to search symbols');
    }
};

module.exports = {
    getStockQuote,
    getTechnicalIndicator,
    getHistoricalAggregates,
    searchSymbols
};