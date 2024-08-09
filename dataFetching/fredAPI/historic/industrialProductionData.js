const axios = require('axios');
const { storeIndustrialProductionData } = require('../../../db/fredAPI/industrialProductionData');

// Series ID for Industrial Production
const seriesID = 'INDPRO';

// Fetch and store Industrial Production data from FRED API
const fetchIndustrialProductionData = async () => {
    try {
        // Get today's date
        const today = new Date().toISOString().split('T')[0];

        const response = await axios.get('https://api.stlouisfed.org/fred/series/observations', {
            params: {
                api_key: process.env.FRED_API_KEY,
                series_id: seriesID,
                file_type: 'json',
                observation_start: '1776-01-01',
                observation_end: today 
            }
        });

        const data = response.data.observations;
        for (const entry of data) {
            const date = entry.date;
            const value = parseFloat(entry.value);

            await storeIndustrialProductionData({ date, value });
        }
    } catch (error) {
        console.error('Error fetching Industrial Production data:', error);
    }
};

module.exports = { fetchIndustrialProductionData };