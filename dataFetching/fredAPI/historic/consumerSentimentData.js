const axios = require('axios');
const { storeConsumerSentimentData } = require('../../../db/fredAPI/consumerSentimentData');

// Series ID for Consumer Sentiment
const seriesID = 'UMCSENT';

// Fetch and store Consumer Sentiment data from FRED API
const fetchConsumerSentimentData = async () => {
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

            await storeConsumerSentimentData({ date, value });
        }
    } catch (error) {
        console.error('Error fetching Consumer Sentiment data:', error);
    }
};

module.exports = { fetchConsumerSentimentData };