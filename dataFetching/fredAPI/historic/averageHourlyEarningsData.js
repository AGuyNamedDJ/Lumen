const axios = require('axios');
const { storeAverageHourlyEarningsData } = require('../../../db/fredAPI/averageHourlyEarningsData');

// Series ID for Average Hourly Earnings
const seriesID = 'CES0500000003';

// Fetch and store Average Hourly Earnings data from FRED API
const fetchAverageHourlyEarningsData = async () => {
    try {
        // Get today's date
        const today = new Date().toISOString().split('T')[0]; // Format: YYYY-MM-DD
        
        // Optional: Specify a start date if you only want to fetch data from a certain point in time
        const startDate = '2000-01-01'; // Example: start fetching from January 1st, 2000

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

            await storeAverageHourlyEarningsData({ date, value });
        }
    } catch (error) {
        console.error('Error fetching Average Hourly Earnings data:', error);
    }
};

module.exports = { fetchAverageHourlyEarningsData };