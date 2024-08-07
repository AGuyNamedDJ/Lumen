const { client } = require('../index');

// Store Interest Rate data
const storeInterestRateData = async ({ date, series_id, value }) => {
    try {
        console.log(`Storing Interest Rate Data: Date - ${date}, Series ID - ${series_id}, Value - ${value}`);
        const query = `
            INSERT INTO interest_rate_data (date, series_id, value)
            VALUES ($1, $2, $3)
            ON CONFLICT (date, series_id) DO NOTHING
            RETURNING *;
        `;
        const values = [date, series_id, value];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error storing Interest Rate data:', error);
    }
};

// Get all Interest Rate data
const getAllInterestRateData = async () => {
    try {
        console.log('Fetching all Interest Rate data');
        const query = 'SELECT * FROM interest_rate_data ORDER BY date DESC;';
        const res = await client.query(query);
        return res.rows;
    } catch (error) {
        console.error('Error fetching Interest Rate data:', error);
    }
};

// Get Interest Rate data by date
const getInterestRateDataByDate = async (date) => {
    try {
        console.log(`Fetching Interest Rate data for date: ${date}`);
        const query = 'SELECT * FROM interest_rate_data WHERE date = $1;';
        const values = [date];
        const res = await client.query(query, values);
        return res.rows;
    } catch (error) {
        console.error('Error fetching Interest Rate data by date:', error);
    }
};

// Update Interest Rate data by date
const updateInterestRateDataByDate = async (date, series_id, value) => {
    try {
        console.log(`Updating Interest Rate data for date: ${date} with series_id: ${series_id} and value: ${value}`);
        const query = `
            UPDATE interest_rate_data
            SET value = $3, updated_at = CURRENT_TIMESTAMP
            WHERE date = $1 AND series_id = $2
            RETURNING *;
        `;
        const values = [date, series_id, value];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error updating Interest Rate data by date:', error);
    }
};

// Delete Interest Rate data by date
const deleteInterestRateDataByDate = async (date, series_id) => {
    try {
        console.log(`Deleting Interest Rate data for date: ${date} and series_id: ${series_id}`);
        const query = 'DELETE FROM interest_rate_data WHERE date = $1 AND series_id = $2 RETURNING *;';
        const values = [date, series_id];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error deleting Interest Rate data by date:', error);
    }
};

module.exports = {
    storeInterestRateData,
    getAllInterestRateData,
    getInterestRateDataByDate,
    updateInterestRateDataByDate,
    deleteInterestRateDataByDate,
};