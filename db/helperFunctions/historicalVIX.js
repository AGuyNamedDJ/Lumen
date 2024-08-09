// Imports
const { client } = require("../index");

// Create Historical VIX Record
async function createHistoricalVIXRecord({ timestamp, open, high, low, close, volume }) {
    try {
        const result = await client.query(`
            INSERT INTO historical_vix (timestamp, open, high, low, close, volume)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING *;
        `, [timestamp, open, high, low, close, volume]);

        return result.rows[0];
    } catch (error) {
        console.error('Error creating historical VIX record:', error);
        throw error;
    }
};

// Get All Historical VIX Records
async function getAllHistoricalVIXRecords() {
    try {
        const result = await client.query('SELECT * FROM historical_vix;');
        return result.rows;
    } catch (error) {
        console.error('Error getting all historical VIX records:', error);
        throw error;
    }
};

// Get Historical VIX Record By ID
async function getHistoricalVIXRecordById(id) {
    try {
        const result = await client.query('SELECT * FROM historical_vix WHERE id = $1;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error getting historical VIX record by ID:', error);
        throw error;
    }
};

// Update Historical VIX Record
async function updateHistoricalVIXRecord(id, fields = {}) {
    const setString = Object.keys(fields).map((key, index) => `"${key}"=$${index + 1}`).join(', ');

    try {
        const result = await client.query(`
            UPDATE historical_vix
            SET ${setString}
            WHERE id = $${Object.keys(fields).length + 1}
            RETURNING *;
        `, [...Object.values(fields), id]);

        return result.rows[0];
    } catch (error) {
        console.error('Error updating historical VIX record:', error);
        throw error;
    }
};

// Delete Historical VIX Record
async function deleteHistoricalVIXRecord(id) {
    try {
        const result = await client.query('DELETE FROM historical_vix WHERE id = $1 RETURNING *;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error deleting historical VIX record:', error);
        throw error;
    }
};

module.exports = {
    createHistoricalVIXRecord,
    getAllHistoricalVIXRecords,
    getHistoricalVIXRecordById,
    updateHistoricalVIXRecord,
    deleteHistoricalVIXRecord
};