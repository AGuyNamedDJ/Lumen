const finnhub = require('finnhub');
require('dotenv').config();
const axios = require('axios');

const FINNHUB_API_KEY='cohasq1r01qrf6b2ivj0cohasq1r01qrf6b2ivjg';

async function getSymbolsByExchange(exchange) {
    const url = `https://finnhub.io/api/v1/stock/symbol?exchange=${exchange}&token=${FINNHUB_API_KEY}`;

    try {
        const response = await axios.get(url);
        if (response.data) {
            console.log('Symbols:', response.data);
            response.data.forEach(symbol => {
                if (symbol.description.includes('VIX')) {
                    console.log('Potential VIX Symbol:', symbol);
                }
            });
        } else {
            console.log('No results found.');
        }
    } catch (error) {
        console.error('Error fetching symbols:', error);
    }
}

// Replace 'US' with the exchange code you want to search
getSymbolsByExchange('US');