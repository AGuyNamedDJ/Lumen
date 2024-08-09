const axios = require('axios');
const { storeInterestRateData } = require('../../../db/fredAPI/interestRateData');


// Fetch and store Interest Rate data from FRED API
const fetchInterestRateData = async (series_id) => {
    try {
        const response = await axios.get('https://api.stlouisfed.org/fred/series/observations', {
            params: {
                api_key: process.env.FRED_API_KEY,
                series_id: series_id,
                file_type: 'json'
            }
        });

        const data = response.data.observations;
        for (const entry of data) {
            const date = entry.date;
            const value = parseFloat(entry.value);

            await storeInterestRateData({ date, series_id, value });
        }
    } catch (error) {
        console.error(`Error fetching Interest Rate data for ${series_id}:`, error);
    }
};

module.exports = {fetchInterestRateData};