const axios = require('axios');

const API_KEY = 'cohasq1r01qrf6b2ivj0cohasq1r01qrf6b2ivjg';
const FINNHUB_URL = 'https://finnhub.io/api/v1/search';

const searchSymbol = async (query) => {
    try {
        const response = await axios.get(FINNHUB_URL, {
            params: { q: query },
            headers: { 'X-Finnhub-Token': API_KEY }
        });
        console.log(response.data);
    } catch (error) {
        console.error('Error fetching symbol data:', error.response ? error.response.data : error.message);
    }
};

searchSymbol('S&P 500');
