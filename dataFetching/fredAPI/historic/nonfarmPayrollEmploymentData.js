const axios = require('axios');
const { storeNonfarmPayrollEmploymentData } = require('../../../db/fredAPI/nonfarmPayrollEmploymentData');

// Series ID for Nonfarm Payroll Employment
const seriesID = 'PAYEMS';

// Fetch and store Nonfarm Payroll Employment data from FRED API
const fetchNonfarmPayrollEmploymentData = async () => {
    try {
        const response = await axios.get('https://api.stlouisfed.org/fred/series/observations', {
            params: {
                api_key: process.env.FRED_API_KEY,
                series_id: seriesID,
                file_type: 'json',
                observation_start: '1940-01-01',  // Start from the earliest available data
                observation_end: new Date().toISOString().split('T')[0]  // Fetch data up until today
            }
        });

        const data = response.data.observations;
        for (const entry of data) {
            const date = entry.date;
            const value = parseFloat(entry.value);

            await storeNonfarmPayrollEmploymentData({ date, value });
        }
    } catch (error) {
        console.error('Error fetching Nonfarm Payroll Employment data:', error);
    }
};

module.exports = { fetchNonfarmPayrollEmploymentData };