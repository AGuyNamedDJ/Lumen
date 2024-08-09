const axios = require('axios');
const { storePCEData } = require('../../../db/fredAPI/pceData');

// Series ID for PCE
const seriesID = 'PCE';

// Fetch and store PCE data from FRED API
const fetchPCEData = async () => {
    try {
        // Get today's date
        const today = new Date().toISOString().split('T')[0];
        const startDate = '2000-01-01'; 

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

            await storePCEData({ date, value });
            console.log(`Storing PCE Data: Date - ${date}, Value - ${value}`);
        }
    } catch (error) {
        console.error('Error fetching PCE data:', error);
    }
};

module.exports = { fetchPCEData };