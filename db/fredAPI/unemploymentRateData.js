const { client } = require('../index');

// Store Unemployment Rate data
const storeUnemploymentRateData = async ({ date, value }) => {
    try {
        console.log(`Storing Unemployment Rate Data: Date - ${date}, Value - ${value}`);
        const query = `
            INSERT INTO unemployment_rate_data (date, value)
            VALUES ($1, $2)
            ON CONFLICT (date) DO NOTHING
            RETURNING *;
        `;
        const values = [date, value];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error storing Unemployment Rate data:', error);
    }
};

// Get all Unemployment Rate data
const getAllUnemploymentRateData = async () => {
    try {
        console.log('Fetching all Unemployment Rate data');
        const query = 'SELECT * FROM unemployment_rate_data ORDER BY date DESC;';
        const res = await client.query(query);
        return res.rows;
    } catch (error) {
        console.error('Error fetching Unemployment Rate data:', error);
    }
};

// Get Unemployment Rate data by date
const getUnemploymentRateDataByDate = async (date) => {
    try {
        console.log(`Fetching Unemployment Rate data for date: ${date}`);
        const query = 'SELECT * FROM unemployment_rate_data WHERE date = $1;';
        const values = [date];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error fetching Unemployment Rate data by date:', error);
    }
};

// Update Unemployment Rate data by date
const updateUnemploymentRateDataByDate = async (date, value) => {
    try {
        console.log(`Updating Unemployment Rate data for date: ${date} with value: ${value}`);
        const query = `
            UPDATE unemployment_rate_data
            SET value = $2, updated_at = CURRENT_TIMESTAMP
            WHERE date = $1
            RETURNING *;
        `;
        const values = [date, value];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error updating Unemployment Rate data by date:', error);
    }
};

// Delete Unemployment Rate data by date
const deleteUnemploymentRateDataByDate = async (date) => {
    try {
        console.log(`Deleting Unemployment Rate data for date: ${date}`);
        const query = 'DELETE FROM unemployment_rate_data WHERE date = $1 RETURNING *;';
        const values = [date];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error deleting Unemployment Rate data by date:', error);
    }
};

module.exports = {
    storeUnemploymentRateData,
    getAllUnemploymentRateData,
    getUnemploymentRateDataByDate,
    updateUnemploymentRateDataByDate,
    deleteUnemploymentRateDataByDate,
};