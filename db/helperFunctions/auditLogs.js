// Imports
const { client } = require("../index");

// Create Audit Log
async function createAuditLog({ action_type, description }) {
    try {
        const result = await client.query(`
            INSERT INTO audit_logs (action_type, description)
            VALUES ($1, $2)
            RETURNING *;
        `, [action_type, description]);

        return result.rows[0];
    } catch (error) {
        console.error('Error creating audit log:', error);
        throw error;
    }
};

// Get All Audit Logs
async function getAllAuditLogs() {
    try {
        const result = await client.query('SELECT * FROM audit_logs;');
        return result.rows;
    } catch (error) {
        console.error('Error getting all audit logs:', error);
        throw error;
    }
};

// Get Audit Log By ID
async function getAuditLogById(id) {
    try {
        const result = await client.query('SELECT * FROM audit_logs WHERE id = $1;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error getting audit log by ID:', error);
        throw error;
    }
};

// Delete Audit Log
async function deleteAuditLog(id) {
    try {
        const result = await client.query('DELETE FROM audit_logs WHERE id = $1 RETURNING *;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error deleting audit log:', error);
        throw error;
    }
};

module.exports = {
    createAuditLog,
    getAllAuditLogs,
    getAuditLogById,
    deleteAuditLog
};