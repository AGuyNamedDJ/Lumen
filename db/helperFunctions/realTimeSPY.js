// Imports
const { client } = require("../index");

// Create Real-Time SPY Record
async function createRealTimeSPYRecord({ timestamp, current_price, volume, conditions }) {
    try {
        const result = await client.query(`
            INSERT INTO real_time_spy (timestamp, current_price, volume, conditions)
            VALUES ($1, $2, $3, $4)
            RETURNING *;
        `, [timestamp, current_price, volume, conditions]);

        return result.rows[0];
    } catch (error) {
        console.error('Error creating real-time SPY record:', error);
        throw error;
    }
};

// Get All Real-Time SPY Records
async function getAllRealTimeSPYRecords() {
    try {
        const result = await client.query('SELECT * FROM real_time_spy ORDER BY timestamp DESC;');
        return result.rows;
    } catch (error) {
        console.error('Error getting all real-time SPY records:', error);
        throw error;
    }
};

// Get Real-Time SPY Record By ID
async function getRealTimeSPYRecordById(id) {
    try {
        const result = await client.query('SELECT * FROM real_time_spy WHERE id = $1;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error getting real-time SPY record by ID:', error);
        throw error;
    }
};

// Update Real-Time SPY Record
async function updateRealTimeSPYRecord(id, fields = {}) {
    const setString = Object.keys(fields).map((key, index) => `"${key}"=$${index + 1}`).join(', ');

    try {
        const result = await client.query(`
            UPDATE real_time_spy
            SET ${setString}
            WHERE id = $${Object.keys(fields).length + 1}
            RETURNING *;
        `, [...Object.values(fields), id]);

        return result.rows[0];
    } catch (error) {
        console.error('Error updating real-time SPY record:', error);
        throw error;
    }
};

// Delete Real-Time SPY Record
async function deleteRealTimeSPYRecord(id) {
    try {
        const result = await client.query('DELETE FROM real_time_spy WHERE id = $1 RETURNING *;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error deleting real-time SPY record:', error);
        throw error;
    }
};

module.exports = {
    createRealTimeSPYRecord,
    getAllRealTimeSPYRecords,
    getRealTimeSPYRecordById,
    updateRealTimeSPYRecord,
    deleteRealTimeSPYRecord
};