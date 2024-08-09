const axios = require('axios');
const { storeLaborForceParticipationData } = require('../../../db/fredAPI/laborForceParticipationRateData');

// Series ID for Labor Force Participation Rate
const seriesID = 'CIVPART';

// Fetch and store Labor Force Participation Rate data from FRED API
const fetchLaborForceParticipationRateData = async () => {
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

            await storeLaborForceParticipationData({ date, value });
        }
    } catch (error) {
        console.error('Error fetching Labor Force Participation Rate data:', error);
    }
};

module.exports = { fetchLaborForceParticipationRateData };