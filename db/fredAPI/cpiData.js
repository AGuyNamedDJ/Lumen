const { client } = require("../index");
require('dotenv').config();

// Store CPI Data
const storeCPIData = async ({ date, value }) => {
    try {
        console.log(`Storing CPI Data: Date - ${date}, Value - ${value}`);
        const query = `
            INSERT INTO cpi_data (date, value)
            VALUES ($1, $2)
            ON CONFLICT (date) DO NOTHING
            RETURNING *;
        `;
        const values = [date, value];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error storing CPI data:', error);
    }
};

// Get All CPI Data
const getAllCPIData = async () => {
    try {
        console.log('Fetching all CPI data');
        const res = await client.query('SELECT * FROM cpi_data ORDER BY date ASC;');
        return res.rows;
    } catch (error) {
        console.error('Error fetching CPI data:', error);
    }
};

// Get CPI Data By Date
const getCPIDataByDate = async (date) => {
    try {
        console.log(`Fetching CPI data for date: ${date}`);
        const res = await client.query('SELECT * FROM cpi_data WHERE date = $1;', [date]);
        return res.rows[0];
    } catch (error) {
        console.error('Error fetching CPI data by date:', error);
    }
};

// Update CPI Data By Date
const updateCPIDataByDate = async (date, value) => {
    try {
        console.log(`Updating CPI data for date: ${date} with value: ${value}`);
        const query = `
            UPDATE cpi_data
            SET value = $2, updated_at = CURRENT_TIMESTAMP
            WHERE date = $1
            RETURNING *;
        `;
        const values = [date, value];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error updating CPI data by date:', error);
    }
};

// Delete CPI Data By Date
const deleteCPIDataByDate = async (date) => {
    try {
        console.log(`Deleting CPI data for date: ${date}`);
        const res = await client.query('DELETE FROM cpi_data WHERE date = $1 RETURNING *;', [date]);
        return res.rows[0];
    } catch (error) {
        console.error('Error deleting CPI data by date:', error);
    }
};

module.exports = {
    storeCPIData,
    getAllCPIData,
    getCPIDataByDate,
    updateCPIDataByDate,
    deleteCPIDataByDate,
};