// Imports
const { client } = require("../index");

// Create Historical Record
async function createHistoricalRecord({ timestamp, open, high, low, close, volume }) {
    try {
        const result = await client.query(`
            INSERT INTO historical_spy (timestamp, open, high, low, close, volume)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING *;
        `, [timestamp, open, high, low, close, volume]);

        return result.rows[0];
    } catch (error) {
        console.error('Error creating historical record:', error);
        throw error;
    }
};

// Get All Historical Records
async function getAllHistoricalRecords() {
    try {
        const result = await client.query('SELECT * FROM historical_spy;');
        return result.rows;
    } catch (error) {
        console.error('Error getting all historical records:', error);
        throw error;
    }
};

// Get Historical Record By ID
async function getHistoricalRecordById(id) {
    try {
        const result = await client.query('SELECT * FROM historical_spy WHERE id = $1;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error getting historical record by ID:', error);
        throw error;
    }
};

// Update Historical Record
async function updateHistoricalRecord(id, fields = {}) {
    const setString = Object.keys(fields).map((key, index) => `"${key}"=$${index + 1}`).join(', ');

    try {
        const result = await client.query(`
            UPDATE historical_spy
            SET ${setString}
            WHERE id = $${Object.keys(fields).length + 1}
            RETURNING *;
        `, [...Object.values(fields), id]);

        return result.rows[0];
    } catch (error) {
        console.error('Error updating historical record:', error);
        throw error;
    }
};

// Delete Historical Record
async function deleteHistoricalRecord(id) {
    try {
        const result = await client.query('DELETE FROM historical_spy WHERE id = $1 RETURNING *;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error deleting historical record:', error);
        throw error;
    }
};

module.exports = {
    createHistoricalRecord,
    getAllHistoricalRecords,
    getHistoricalRecordById,
    updateHistoricalRecord,
    deleteHistoricalRecord
};