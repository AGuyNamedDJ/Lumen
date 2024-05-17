// Imports
const { client } = require("../index");

// Create a record in detailed_historical_spx
async function createDetailedRecord({ timestamp, price, volume }) {
    try {
        const result = await client.query(`
            INSERT INTO detailed_historical_spx (timestamp, price, volume)
            VALUES ($1, $2, $3)
            RETURNING *;
        `, [timestamp, price, volume]);

        return result.rows[0];
    } catch (error) {
        console.error('Error creating detailed record:', error);
        throw error;
    }
}


// Get all records from detailed_historical_spx
async function getAllDetailedRecords() {
    try {
        const result = await client.query('SELECT * FROM detailed_historical_spx ORDER BY timestamp ASC;');
        return result.rows;
    } catch (error) {
        console.error('Error fetching all detailed records:', error);
        throw error;
    }
};

// Get a record by ID from detailed_historical_spx
async function getDetailedRecordById(id) {
    try {
        const result = await client.query('SELECT * FROM detailed_historical_spx WHERE id = $1;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error fetching detailed record by ID:', error);
        throw error;
    }
};

// Update a record in detailed_historical_spx
async function updateDetailedRecord(id, fields = {}) {
    const setString = Object.keys(fields).map((key, index) => `"${key}"=$${index + 1}`).join(', ');

    if (!setString.length) {
        return;
    }

    try {
        const result = await client.query(`
            UPDATE detailed_historical_spx
            SET ${setString}
            WHERE id = $${Object.keys(fields).length + 1}
            RETURNING *;
        `, [...Object.values(fields), id]);

        return result.rows[0];
    } catch (error) {
        console.error('Error updating detailed record:', error);
        throw error;
    }
}

// Delete a record from detailed_historical_spx
async function deleteDetailedRecord(id) {
    try {
        const result = await client.query('DELETE FROM detailed_historical_spx WHERE id = $1 RETURNING *;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error deleting detailed record:', error);
        throw error;
    }
};

module.exports = {
    createDetailedRecord,
    getAllDetailedRecords,
    getDetailedRecordById,
    updateDetailedRecord,
    deleteDetailedRecord,
};