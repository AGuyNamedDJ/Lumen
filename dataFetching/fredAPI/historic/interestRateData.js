const axios = require('axios');
const { storeInterestRateData } = require('../../../db/fredAPI/interestRateData');

const seriesIDs = ['FEDFUNDS', 'DFF', 'MPRIME', 'GS10', 'TB3MS'];

// Fetch and store Interest Rate data from FRED API
const fetchInterestRateData = async (series_id) => {
    try {
        if (!series_id) throw new Error('Series ID is undefined');

        // Get today's date
        const today = new Date().toISOString().split('T')[0];

        const response = await axios.get('https://api.stlouisfed.org/fred/series/observations', {
            params: {
                api_key: process.env.FRED_API_KEY,
                series_id: series_id,
                file_type: 'json',
                observation_start: '1776-01-01', 
                observation_end: today 
            }
        });

        const data = response.data.observations;
        for (const entry of data) {
            const date = entry.date;
            const value = parseFloat(entry.value);

            await storeInterestRateData({ date, series_id, value });
            console.log(`Stored Interest Rate Data: Date - ${date}, Value - ${value}, Series ID - ${series_id}`);
        }
    } catch (error) {
        console.error(`Error fetching Interest Rate data for ${series_id}:`, error);
    }
};

// Fetch data for all series IDs
const fetchAllInterestRateData = async () => {
    for (const id of seriesIDs) {
        await fetchInterestRateData(id);
    }
};

module.exports = { fetchAllInterestRateData };