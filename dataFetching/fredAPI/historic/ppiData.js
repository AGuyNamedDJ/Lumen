const axios = require('axios');
const { storePPIData } = require('../../../db/fredAPI/ppiData');

// Series ID for PPI
const seriesID = 'PPIACO';

// Fetch and store PPI data from FRED API
const fetchPPIData = async () => {
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
            await storePPIData({ date, value });
        }
    } catch (error) {
        console.error('Error fetching PPI data:', error);
    }
};

module.exports = { fetchPPIData };