// Imports
const { client } = require("../index");

// Create Real-Time SPX Record
async function createRealTimeSPXRecord({ timestamp, current_price, volume, conditions }) {
    try {
        const result = await client.query(`
            INSERT INTO real_time_spx (timestamp, current_price, volume, conditions)
            VALUES ($1, $2, $3, $4)
            RETURNING *;
        `, [timestamp, current_price, volume, conditions]);

        return result.rows[0];
    } catch (error) {
        console.error('Error creating real-time SPX record:', error);
        throw error;
    }
};

// Get All Real-Time SPX Records
async function getAllRealTimeSPXRecords() {
    try {
        const result = await client.query('SELECT * FROM real_time_spx ORDER BY timestamp DESC;');
        return result.rows;
    } catch (error) {
        console.error('Error getting all real-time SPX records:', error);
        throw error;
    }
};

// Get Real-Time SPX Record By ID
async function getRealTimeSPXRecordById(id) {
    try {
        const result = await client.query('SELECT * FROM real_time_spx WHERE id = $1;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error getting real-time SPX record by ID:', error);
        throw error;
    }
};

// Update Real-Time SPX Record
async function updateRealTimeSPXRecord(id, fields = {}) {
    const setString = Object.keys(fields).map((key, index) => `"${key}"=$${index + 1}`).join(', ');

    try {
        const result = await client.query(`
            UPDATE real_time_spx
            SET ${setString}
            WHERE id = $${Object.keys(fields).length + 1}
            RETURNING *;
        `, [...Object.values(fields), id]);

        return result.rows[0];
    } catch (error) {
        console.error('Error updating real-time SPX record:', error);
        throw error;
    }
};

// Delete Real-Time SPX Record
async function deleteRealTimeSPXRecord(id) {
    try {
        const result = await client.query('DELETE FROM real_time_spx WHERE id = $1 RETURNING *;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error deleting real-time SPX record:', error);
        throw error;
    }
};

module.exports = {
    createRealTimeSPXRecord,
    getAllRealTimeSPXRecords,
    getRealTimeSPXRecordById,
    updateRealTimeSPXRecord,
    deleteRealTimeSPXRecord
};