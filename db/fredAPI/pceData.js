const { client } = require('../index');

// Store PCE data
const storePCEData = async ({ date, value }) => {
    try {
        console.log(`Storing PCE Data: Date - ${date}, Value - ${value}`);
        const query = `
            INSERT INTO personal_consumption_expenditures (date, value)
            VALUES ($1, $2)
            ON CONFLICT (date) DO NOTHING
            RETURNING *;
        `;
        const values = [date, value];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error storing PCE data:', error);
    }
};

// Get all PCE data
const getAllPCEData = async () => {
    try {
        console.log('Fetching all PCE data');
        const query = 'SELECT * FROM personal_consumption_expenditures ORDER BY date DESC;';
        const res = await client.query(query);
        return res.rows;
    } catch (error) {
        console.error('Error fetching PCE data:', error);
    }
};

// Get PCE data by date
const getPCEDataByDate = async (date) => {
    try {
        console.log(`Fetching PCE data for date: ${date}`);
        const query = 'SELECT * FROM personal_consumption_expenditures WHERE date = $1;';
        const values = [date];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error fetching PCE data by date:', error);
    }
};

// Update PCE data by date
const updatePCEDataByDate = async (date, value) => {
    try {
        console.log(`Updating PCE data for date: ${date} with value: ${value}`);
        const query = `
            UPDATE personal_consumption_expenditures
            SET value = $2, updated_at = CURRENT_TIMESTAMP
            WHERE date = $1
            RETURNING *;
        `;
        const values = [date, value];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error updating PCE data by date:', error);
    }
};

// Delete PCE data by date
const deletePCEDataByDate = async (date) => {
    try {
        console.log(`Deleting PCE data for date: ${date}`);
        const query = 'DELETE FROM personal_consumption_expenditures WHERE date = $1 RETURNING *;';
        const values = [date];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error deleting PCE data by date:', error);
    }
};

module.exports = {
    storePCEData,
    getAllPCEData,
    getPCEDataByDate,
    updatePCEDataByDate,
    deletePCEDataByDate,
};