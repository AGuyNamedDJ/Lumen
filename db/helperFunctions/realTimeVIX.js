// Imports
const { client } = require("../index");

// Create Real-Time VIX Record
async function createRealTimeVIXRecord({ timestamp, current_price, volume, conditions, open, high, low, close }) {
    try {
        const result = await client.query(`
            INSERT INTO real_time_vix (timestamp, current_price, volume, conditions, open, high, low, close)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING *;
        `, [timestamp, current_price, volume, conditions, open, high, low, close]);

        return result.rows[0];
    } catch (error) {
        console.error('Error creating real-time VIX record:', error);
        throw error;
    }
};

// Get All Real-Time VIX Records
async function getAllRealTimeVIXRecords() {
    try {
        const result = await client.query('SELECT * FROM real_time_vix ORDER BY timestamp DESC;');
        return result.rows;
    } catch (error) {
        console.error('Error getting all real-time VIX records:', error);
        throw error;
    }
};

// Get Real-Time VIX Record By ID
async function getRealTimeVIXRecordById(id) {
    try {
        const result = await client.query('SELECT * FROM real_time_vix WHERE id = $1;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error getting real-time VIX record by ID:', error);
        throw error;
    }
};

// Update Real-Time VIX Record
async function updateRealTimeVIXRecord(id, fields = {}) {
    const setString = Object.keys(fields).map((key, index) => `"${key}"=$${index + 1}`).join(', ');

    try {
        const result = await client.query(`
            UPDATE real_time_vix
            SET ${setString}
            WHERE id = $${Object.keys(fields).length + 1}
            RETURNING *;
        `, [...Object.values(fields), id]);

        return result.rows[0];
    } catch (error) {
        console.error('Error updating real-time VIX record:', error);
        throw error;
    }
};

// Delete Real-Time VIX Record
async function deleteRealTimeVIXRecord(id) {
    try {
        const result = await client.query('DELETE FROM real_time_vix WHERE id = $1 RETURNING *;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error deleting real-time VIX record:', error);
        throw error;
    }
};

module.exports = {
    createRealTimeVIXRecord,
    getAllRealTimeVIXRecords,
    getRealTimeVIXRecordById,
    updateRealTimeVIXRecord,
    deleteRealTimeVIXRecord
};