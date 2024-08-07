const { client } = require('../index');

// Store GDP data
const storeGDPData = async ({ date, value }) => {
    try {
        console.log(`Storing GDP Data: Date - ${date}, Value - ${value}`);
        const query = `
            INSERT INTO gdp_data (date, value)
            VALUES ($1, $2)
            ON CONFLICT (date) DO NOTHING
            RETURNING *;
        `;
        const values = [date, value];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error storing GDP data:', error);
    }
};

// Get all GDP data
const getAllGDPData = async () => {
    try {
        console.log('Fetching all GDP data');
        const query = 'SELECT * FROM gdp_data ORDER BY date DESC;';
        const res = await client.query(query);
        return res.rows;
    } catch (error) {
        console.error('Error fetching GDP data:', error);
    }
};

// Get GDP data by date
const getGDPDataByDate = async (date) => {
    try {
        console.log(`Fetching GDP data for date: ${date}`);
        const query = 'SELECT * FROM gdp_data WHERE date = $1;';
        const values = [date];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error fetching GDP data by date:', error);
    }
};

// Update GDP data by date
const updateGDPDataByDate = async (date, value) => {
    try {
        console.log(`Updating GDP data for date: ${date} with value: ${value}`);
        const query = `
            UPDATE gdp_data
            SET value = $2, updated_at = CURRENT_TIMESTAMP
            WHERE date = $1
            RETURNING *;
        `;
        const values = [date, value];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error updating GDP data by date:', error);
    }
};

// Delete GDP data by date
const deleteGDPDataByDate = async (date) => {
    try {
        console.log(`Deleting GDP data for date: ${date}`);
        const query = 'DELETE FROM gdp_data WHERE date = $1 RETURNING *;';
        const values = [date];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error deleting GDP data by date:', error);
    }
};

module.exports = {
    storeGDPData,
    getAllGDPData,
    getGDPDataByDate,
    updateGDPDataByDate,
    deleteGDPDataByDate,
};