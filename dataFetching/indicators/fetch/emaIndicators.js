const axios = require('axios');
const { storeEMAData } = require('../../../db/indicators/emaIndicators');
const POLYGON_API_KEY = process.env.POLYGON_API_KEY;

console.log("Loading emaIndicators.js...");
console.log("Current Directory:", __dirname);

async function fetchEMAData(symbol, window, series_type) {
    try {
        const timespan = 'day';  // Standardize to 'day'
        const endpoint = `/v1/indicators/ema/${symbol}`;
        const params = { 
            timespan, 
            window, 
            series_type, 
            apiKey: POLYGON_API_KEY 
        };

        // Log the request parameters
        console.log(`Fetching EMA data with parameters: ${JSON.stringify(params)}`);

        // Make the API request directly using axios
        const response = await axios.get(`https://api.polygon.io${endpoint}`, { params });

        // Log the received data
        const indicatorData = response.data;
        console.log('Received data from Polygon API:', JSON.stringify(indicatorData, null, 2));

        // Check if the response format is correct
        if (indicatorData && indicatorData.results && Array.isArray(indicatorData.results.values)) {
            const formattedData = indicatorData.results.values.map(dataPoint => ({
                symbol,
                period: window,
                timespan,
                timestamp: dataPoint.timestamp || dataPoint[0],
                value: dataPoint.value || dataPoint[1]
            }));

            // Log the formatted data before storing
            console.log('Formatted EMA data to store:', JSON.stringify(formattedData, null, 2));

            // Store the EMA data
            await storeEMAData(formattedData);
            console.log(`EMA data for ${symbol} (${timespan} - ${window}) fetched and stored successfully.`);
        } else {
            console.error('Unexpected response format from Polygon API:', indicatorData);
        }
    } catch (error) {
        console.error('Error fetching and storing EMA data:', error);
    }
};

module.exports = { fetchEMAData };