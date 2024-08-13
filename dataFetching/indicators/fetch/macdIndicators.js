const axios = require('axios');
const { storeMACDData } = require('../../../db/indicators/macdIndicators');
const POLYGON_API_KEY = process.env.POLYGON_API_KEY;

async function fetchMACDData(symbol, timespan, series_type, fast_period, slow_period, signal_period) {
    try {
        const validSeriesTypes = ["close", "open", "high", "low"]; // Add validation for series_type

        // Validate series_type before making the request
        if (!validSeriesTypes.includes(series_type)) {
            throw new Error(`Invalid series_type: ${series_type}. Must be one of ${validSeriesTypes.join(", ")}`);
        }

        const endpoint = `/v1/indicators/macd/${symbol}`;
        
        const params = { 
            timespan, 
            series_type, 
            fast_period, 
            slow_period, 
            signal_period,
            apiKey: POLYGON_API_KEY
        };

        // Make the API request directly using axios
        const response = await axios.get(`https://api.polygon.io${endpoint}`, { params });

        const indicatorData = response.data;

        if (indicatorData && Array.isArray(indicatorData.results)) {
            const formattedData = indicatorData.results.map(dataPoint => ({
                symbol,
                timespan,
                timestamp: dataPoint.timestamp || dataPoint[0],
                macd: dataPoint.macd || dataPoint[1],  // The MACD line value
                signal: dataPoint.signal || dataPoint[2],  // The signal line value
                histogram: dataPoint.histogram || dataPoint[3]  // The MACD histogram value
            }));

            await storeMACDData(formattedData);
            console.log(`MACD data for ${symbol} (${timespan}) fetched and stored successfully.`);
        } else {
            console.error(`Unexpected response format from Polygon API for ${symbol} (${timespan})`);
            console.error(`Received data:`, JSON.stringify(indicatorData, null, 2)); 
        }
    } catch (error) {
        console.error('Error fetching and storing MACD data:', error);
    }
};

module.exports = { fetchMACDData };