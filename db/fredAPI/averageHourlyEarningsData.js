const { client } = require('../index');

// Store Average Hourly Earnings data
const storeAverageHourlyEarningsData = async ({ date, value }) => {
    try {
        console.log(`Storing Average Hourly Earnings Data: Date - ${date}, Value - ${value}`);
        const query = `
            INSERT INTO average_hourly_earnings_data (date, value)
            VALUES ($1, $2)
            ON CONFLICT (date) DO NOTHING
            RETURNING *;
        `;
        const values = [date, value];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error storing Average Hourly Earnings data:', error);
    }
};

// Get all Average Hourly Earnings data
const getAllAverageHourlyEarningsData = async () => {
    try {
        console.log('Fetching all Average Hourly Earnings data');
        const query = 'SELECT * FROM average_hourly_earnings_data ORDER BY date DESC;';
        const res = await client.query(query);
        return res.rows;
    } catch (error) {
        console.error('Error fetching Average Hourly Earnings data:', error);
    }
};

// Get Average Hourly Earnings data by date
const getAverageHourlyEarningsDataByDate = async (date) => {
    try {
        console.log(`Fetching Average Hourly Earnings data for date: ${date}`);
        const query = 'SELECT * FROM average_hourly_earnings_data WHERE date = $1;';
        const values = [date];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error fetching Average Hourly Earnings data by date:', error);
    }
};

// Update Average Hourly Earnings data by date
const updateAverageHourlyEarningsDataByDate = async (date, value) => {
    try {
        console.log(`Updating Average Hourly Earnings data for date: ${date} with value: ${value}`);
        const query = `
            UPDATE average_hourly_earnings_data
            SET value = $2, updated_at = CURRENT_TIMESTAMP
            WHERE date = $1
            RETURNING *;
        `;
        const values = [date, value];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error updating Average Hourly Earnings data by date:', error);
    }
};

// Delete Average Hourly Earnings data by date
const deleteAverageHourlyEarningsDataByDate = async (date) => {
    try {
        console.log(`Deleting Average Hourly Earnings data for date: ${date}`);
        const query = 'DELETE FROM average_hourly_earnings_data WHERE date = $1 RETURNING *;';
        const values = [date];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error deleting Average Hourly Earnings data by date:', error);
    }
};

module.exports = {
    storeAverageHourlyEarningsData,
    getAllAverageHourlyEarningsData,
    getAverageHourlyEarningsDataByDate,
    updateAverageHourlyEarningsDataByDate,
    deleteAverageHourlyEarningsDataByDate,
};