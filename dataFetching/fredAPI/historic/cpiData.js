const axios = require('axios');
const { storeCPIData } = require('../../../db/fredAPI/cpiData');

// Series ID for Core Inflation
const seriesID = 'CPIAUCSL';

// Fetch and store CPI data from FRED API
const fetchCPIData = async () => {
    try {
        const today = new Date().toISOString().split('T')[0]; // Format: YYYY-MM-DD
        const startDate = '1947-01-01'; // Fetching data from January 1st, 1947
        
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
            await storeCPIData({ date, value });
        }
        console.log('CPI data fetched and stored successfully.');
    } catch (error) {
        console.error('Error fetching CPI data:', error);
    }
};

module.exports = { fetchCPIData };