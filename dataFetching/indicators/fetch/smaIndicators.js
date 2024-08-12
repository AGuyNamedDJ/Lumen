const axios = require('axios');
const { storeSMAData } = require('../../../db/indicators/smaIndicators');
const POLYGON_API_KEY = process.env.POLYGON_API_KEY;
const { polygonApiRequest } = require('../../../utils/polygonApiClient');

async function fetchAndStoreSMAData(symbol, window, series_type) {
    try {
        const timespan = 'day';  // Standardize to 'day'
        const endpoint = `/v1/indicators/sma/${symbol}`;
        
        const params = {
            timespan: timespan,
            window: window,
            series_type: series_type,
            apiKey: POLYGON_API_KEY
        };

        const indicatorData = await polygonApiRequest(endpoint, params);

        if (indicatorData && Array.isArray(indicatorData.results)) {
            const formattedData = indicatorData.results.map(dataPoint => ({
                symbol,
                period: window,
                timespan,
                timestamp: dataPoint.timestamp || dataPoint[0],
                value: dataPoint.value || dataPoint[1]
            }));

            await storeSMAData(formattedData);
            console.log(`SMA data for ${symbol} (${timespan}, window: ${window}) stored successfully.`);
        } else {
            console.error(`Unexpected response format from Polygon API for ${symbol} (${timespan}, window: ${window})`);
            console.error(`Received data:`, JSON.stringify(indicatorData, null, 2));  // Log the unexpected data structure
        }
    } catch (error) {
        console.error(`Failed to fetch and store SMA data for ${symbol} (${timespan}, window: ${window})`, error);
    }
};

module.exports = { fetchAndStoreSMAData };