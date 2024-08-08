const { client } = require('../index');

// Store Industrial Production data
const storeIndustrialProductionData = async ({ date, value }) => {
    try {
        console.log(`Storing Industrial Production Data: Date - ${date}, Value - ${value}`);
        const query = `
            INSERT INTO industrial_production_data (date, value)
            VALUES ($1, $2)
            ON CONFLICT (date) DO NOTHING
            RETURNING *;
        `;
        const values = [date, value];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error storing Industrial Production data:', error);
    }
};

// Get all Industrial Production data
const getAllIndustrialProductionData = async () => {
    try {
        console.log('Fetching all Industrial Production data');
        const query = 'SELECT * FROM industrial_production_data ORDER BY date DESC;';
        const res = await client.query(query);
        return res.rows;
    } catch (error) {
        console.error('Error fetching Industrial Production data:', error);
    }
};

// Get Industrial Production data by date
const getIndustrialProductionDataByDate = async (date) => {
    try {
        console.log(`Fetching Industrial Production data for date: ${date}`);
        const query = 'SELECT * FROM industrial_production_data WHERE date = $1;';
        const values = [date];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error fetching Industrial Production data by date:', error);
    }
};

// Update Industrial Production data by date
const updateIndustrialProductionDataByDate = async (date, value) => {
    try {
        console.log(`Updating Industrial Production data for date: ${date} with value: ${value}`);
        const query = `
            UPDATE industrial_production_data
            SET value = $2, updated_at = CURRENT_TIMESTAMP
            WHERE date = $1
            RETURNING *;
        `;
        const values = [date, value];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error updating Industrial Production data by date:', error);
    }
};

// Delete Industrial Production data by date
const deleteIndustrialProductionDataByDate = async (date) => {
    try {
        console.log(`Deleting Industrial Production data for date: ${date}`);
        const query = 'DELETE FROM industrial_production_data WHERE date = $1 RETURNING *;';
        const values = [date];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error deleting Industrial Production data by date:', error);
    }
};

module.exports = {
    storeIndustrialProductionData,
    getAllIndustrialProductionData,
    getIndustrialProductionDataByDate,
    updateIndustrialProductionDataByDate,
    deleteIndustrialProductionDataByDate,
};