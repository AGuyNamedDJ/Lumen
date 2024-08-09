const axios = require('axios');
const { storeConsumerConfidenceData } = require('../../../db/fredAPI/consumerConfidenceData');

// Series ID for Average Hourly Earnings
const seriesID = 'CSCICP03USM665S';

// Fetch and store Consumer Confidence data from FRED API
const fetchConsumerConfidenceData = async () => {
    try {
        const today = new Date().toISOString().split('T')[0];
        const startDate = '2000-01-01'; 

        const response = await axios.get('https://api.stlouisfed.org/fred/series/observations', {
            params: {
                api_key: process.env.FRED_API_KEY,
                series_id: 'CSCICP03USM665S',
                file_type: 'json',
                observation_start: startDate,
                observation_end: today
            }
        });

        const data = response.data.observations;
        for (const entry of data) {
            const date = entry.date;
            const value = parseFloat(entry.value);

            await storeConsumerConfidenceData({ date, value });
        }
    } catch (error) {
        console.error('Error fetching Consumer Confidence data:', error);
    }
};

module.exports = { fetchConsumerConfidenceData };