const { client } = require('../index');

// Store PPI Data
const storePPIData = async ({ date, value }) => {
    try {
        console.log(`Storing PPI Data: Date - ${date}, Value - ${value}`);
        const query = `
            INSERT INTO ppi_data (date, value)
            VALUES ($1, $2)
            ON CONFLICT (date) DO NOTHING
            RETURNING *;
        `;
        const values = [date, value];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error storing PPI data:', error);
        throw error;
    }
};

// Get All PPI Data
const getAllPPIData = async () => {
    try {
        console.log('Fetching all PPI data');
        const res = await client.query('SELECT * FROM ppi_data ORDER BY date DESC');
        return res.rows;
    } catch (error) {
        console.error('Error fetching PPI data:', error);
        throw error;
    }
};

// Get PPI Data by Date
const getPPIDataByDate = async (date) => {
    try {
        console.log(`Fetching PPI data for date: ${date}`);
        const res = await client.query('SELECT * FROM ppi_data WHERE date = $1', [date]);
        return res.rows[0];
    } catch (error) {
        console.error('Error fetching PPI data by date:', error);
        throw error;
    }
};

// Update PPI Data by Date
const updatePPIDataByDate = async (date, value) => {
    try {
        console.log(`Updating PPI data for date: ${date} with value: ${value}`);
        const query = `
            UPDATE ppi_data
            SET value = $2, updated_at = CURRENT_TIMESTAMP
            WHERE date = $1
            RETURNING *;
        `;
        const res = await client.query(query, [date, value]);
        return res.rows[0];
    } catch (error) {
        console.error('Error updating PPI data by date:', error);
        throw error;
    }
};

// Delete PPI Data by Date
const deletePPIDataByDate = async (date) => {
    try {
        console.log(`Deleting PPI data for date: ${date}`);
        const res = await client.query('DELETE FROM ppi_data WHERE date = $1 RETURNING *', [date]);
        return res.rows[0];
    } catch (error) {
        console.error('Error deleting PPI data by date:', error);
        throw error;
    }
};

module.exports = {
    storePPIData,
    getAllPPIData,
    getPPIDataByDate,
    updatePPIDataByDate,
    deletePPIDataByDate,
};