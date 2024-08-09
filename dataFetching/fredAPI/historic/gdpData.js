const axios = require('axios');
const { storeGDPData } = require('../../../db/fredAPI/gdpData');

// Series ID for Core Inflation
const seriesID = 'GDP';

// Fetch and store GDP data from FRED API
const fetchGDPData = async () => {
    try {
        const today = new Date().toISOString().split('T')[0]; 
        const startDate = '1947-01-01';
        
        const response = await axios.get('https://api.stlouisfed.org/fred/series/observations', {
            params: {
                api_key: process.env.FRED_API_KEY,
                series_id: seriesID,
                file_type: 'json',
                observation_start: startDate,
                observation_end: today
            }
        });

        const data = response.data.observations;
        for (const entry of data) {
            const date = entry.date;
            const value = parseFloat(entry.value);
            await storeGDPData({ date, value });
        }
        console.log('GDP data fetched and stored successfully.');
    } catch (error) {
        console.error('Error fetching GDP data:', error);
    }
};

module.exports = { fetchGDPData };