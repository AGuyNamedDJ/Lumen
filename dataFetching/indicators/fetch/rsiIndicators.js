const axios = require('axios');
const { storeRSIData } = require('../../../db/indicators/rsiIndicators');
const POLYGON_API_KEY = process.env.POLYGON_API_KEY;

async function fetchAndStoreRSIData(symbol, window, series_type) {
    try {
        const timespan = 'day';  // Standardize to 'day'
        const endpoint = `/v1/indicators/rsi/${symbol}`;
        
        const params = {
            timespan: timespan,
            window: window,
            series_type: series_type,
            apiKey: POLYGON_API_KEY
        };

        console.log(`Fetching RSI data with parameters:`, params);

        // Make the API request directly using axios
        const response = await axios.get(`https://api.polygon.io${endpoint}`, { params });

        const indicatorData = response.data;

        if (indicatorData && Array.isArray(indicatorData.results)) {
            const formattedData = indicatorData.results.map(dataPoint => ({
                symbol,
                period: window,
                timespan,
                timestamp: dataPoint.timestamp || dataPoint[0],
                value: dataPoint.value || dataPoint[1]
            }));

            await storeRSIData(formattedData);
            console.log(`RSI data for ${symbol} (${timespan}, window: ${window}) stored successfully.`);
        } else {
            console.error(`Unexpected response format from Polygon API for ${symbol} (${timespan}, window: ${window})`);
            console.error(`Received data:`, JSON.stringify(indicatorData, null, 2)); 
        }
    } catch (error) {
        console.error(`Failed to fetch and store RSI data for ${symbol} (${timespan}, window: ${window})`, error);
    }
};

module.exports = { fetchAndStoreRSIData };