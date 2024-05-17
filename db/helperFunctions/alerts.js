// Imports
const { client } = require("../index");

// Create Alert
async function createAlert({ user_id, trade_id, strategy_id, message, alert_type }) {
    try {
        const result = await client.query(`
            INSERT INTO alerts (user_id, trade_id, strategy_id, message, alert_type)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING *;
        `, [user_id, trade_id, strategy_id, message, alert_type]);

        return result.rows[0];
    } catch (error) {
        console.error('Error creating alert:', error);
        throw error;
    }
};

// Get All Alerts
async function getAllAlerts() {
    try {
        const result = await client.query('SELECT * FROM alerts;');
        return result.rows;
    } catch (error) {
        console.error('Error getting all alerts:', error);
        throw error;
    }
};

// Get Alert By ID
async function getAlertById(id) {
    try {
        const result = await client.query('SELECT * FROM alerts WHERE id = $1;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error getting alert by ID:', error);
        throw error;
    }
};

// Update Alert
async function updateAlert(id, fields = {}) {
    const setString = Object.keys(fields).map((key, index) => `"${key}"=$${index + 1}`).join(', ');

    try {
        const result = await client.query(`
            UPDATE alerts
            SET ${setString}
            WHERE id = $${Object.keys(fields).length + 1}
            RETURNING *;
        `, [...Object.values(fields), id]);

        return result.rows[0];
    } catch (error) {
        console.error('Error updating alert:', error);
        throw error;
    }
};

// Delete Alert
async function deleteAlert(id) {
    try {
        const result = await client.query('DELETE FROM alerts WHERE id = $1 RETURNING *;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error deleting alert:', error);
        throw error;
    }
};

module.exports = {
    createAlert,
    getAllAlerts,
    getAlertById,
    updateAlert,
    deleteAlert
};