const { client } = require('../index');

// Store Core Inflation data
const storeCoreInflationData = async ({ date, value }) => {
    try {
        console.log(`Storing Core Inflation Data: Date - ${date}, Value - ${value}`);
        const query = `
            INSERT INTO core_inflation_data (date, value)
            VALUES ($1, $2)
            ON CONFLICT (date) DO NOTHING
            RETURNING *;
        `;
        const values = [date, value];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error storing Core Inflation data:', error);
    }
};

// Get all Core Inflation data
const getAllCoreInflationData = async () => {
    try {
        console.log('Fetching all Core Inflation data');
        const query = 'SELECT * FROM core_inflation_data ORDER BY date DESC;';
        const res = await client.query(query);
        return res.rows;
    } catch (error) {
        console.error('Error fetching Core Inflation data:', error);
    }
};

// Get Core Inflation data by date
const getCoreInflationDataByDate = async (date) => {
    try {
        console.log(`Fetching Core Inflation data for date: ${date}`);
        const query = 'SELECT * FROM core_inflation_data WHERE date = $1;';
        const values = [date];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error fetching Core Inflation data by date:', error);
    }
};

// Update Core Inflation data by date
const updateCoreInflationDataByDate = async (date, value) => {
    try {
        console.log(`Updating Core Inflation data for date: ${date} with value: ${value}`);
        const query = `
            UPDATE core_inflation_data
            SET value = $2, updated_at = CURRENT_TIMESTAMP
            WHERE date = $1
            RETURNING *;
        `;
        const values = [date, value];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error updating Core Inflation data by date:', error);
    }
};

// Delete Core Inflation data by date
const deleteCoreInflationDataByDate = async (date) => {
    try {
        console.log(`Deleting Core Inflation data for date: ${date}`);
        const query = 'DELETE FROM core_inflation_data WHERE date = $1 RETURNING *;';
        const values = [date];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error deleting Core Inflation data by date:', error);
    }
};

module.exports = {
    storeCoreInflationData,
    getAllCoreInflationData,
    getCoreInflationDataByDate,
    updateCoreInflationDataByDate,
    deleteCoreInflationDataByDate,
};