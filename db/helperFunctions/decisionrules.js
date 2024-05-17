// Imports
const { client } = require("../index");

// Create Decision Rule
async function createDecisionRule({ strategy_id, parameter_name, value, description }) {
    try {
        const result = await client.query(`
            INSERT INTO decision_rules (strategy_id, parameter_name, value, description)
            VALUES ($1, $2, $3, $4)
            RETURNING *;
        `, [strategy_id, parameter_name, value, description]);

        return result.rows[0];
    } catch (error) {
        console.error('Error creating decision rule:', error);
        throw error;
    }
};

// Get All Decision Rules
async function getAllDecisionRules() {
    try {
        const result = await client.query('SELECT * FROM decision_rules;');
        return result.rows;
    } catch (error) {
        console.error('Error getting all decision rules:', error);
        throw error;
    }
};

// Get Decision Rule By ID
async function getDecisionRuleById(id) {
    try {
        const result = await client.query('SELECT * FROM decision_rules WHERE id = $1;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error getting decision rule by ID:', error);
        throw error;
    }
};

// Update Decision Rule
async function updateDecisionRule(id, fields = {}) {
    const setString = Object.keys(fields).map((key, index) => `"${key}"=$${index + 1}`).join(', ');

    try {
        const result = await client.query(`
            UPDATE decision_rules
            SET ${setString}
            WHERE id = $${Object.keys(fields).length + 1}
            RETURNING *;
        `, [...Object.values(fields), id]);

        return result.rows[0];
    } catch (error) {
        console.error('Error updating decision rule:', error);
        throw error;
    }
};

// Delete Decision Rule
async function deleteDecisionRule(id) {
    try {
        const result = await client.query('DELETE FROM decision_rules WHERE id = $1 RETURNING *;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error deleting decision rule:', error);
        throw error;
    }
};

module.exports = {
    createDecisionRule,
    getAllDecisionRules,
    getDecisionRuleById,
    updateDecisionRule,
    deleteDecisionRule
};