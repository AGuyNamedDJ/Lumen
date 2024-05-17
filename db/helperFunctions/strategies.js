// Imports
const { client } = require("../index");

// Create Strategy
async function createStrategy({ name, description }) {
    try {
        const result = await client.query(`
            INSERT INTO strategies (name, description)
            VALUES ($1, $2)
            RETURNING *;
        `, [name, description]);

        return result.rows[0];
    } catch (error) {
        console.error('Error creating strategy:', error);
        throw error;
    }
};

// Get All Strategies
async function getAllStrategies() {
    try {
        const result = await client.query('SELECT * FROM strategies;');
        return result.rows;
    } catch (error) {
        console.error('Error getting all strategies:', error);
        throw error;
    }
};

// Get Strategy By Name
async function getStrategyByName(name) {
    try {
        const result = await client.query('SELECT * FROM strategies WHERE name = $1;', [name]);
        return result.rows[0];
    } catch (error) {
        console.error('Error getting strategy by name:', error);
        throw error;
    }
};

module.exports = {
    createStrategy,
    getAllStrategies,
    getStrategyById,
    updateStrategy,
    deleteStrategy,
    getStrategyByName // Export the new function
};


// Get Strategy By ID
async function getStrategyById(id) {
    try {
        const result = await client.query('SELECT * FROM strategies WHERE id = $1;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error getting strategy by ID:', error);
        throw error;
    }
};

// Update Strategy
async function updateStrategy(id, fields = {}) {
    const setString = Object.keys(fields).map((key, index) => `"${key}"=$${index + 1}`).join(', ');

    try {
        const result = await client.query(`
            UPDATE strategies
            SET ${setString}
            WHERE id = $${Object.keys(fields).length + 1}
            RETURNING *;
        `, [...Object.values(fields), id]);

        return result.rows[0];
    } catch (error) {
        console.error('Error updating strategy:', error);
        throw error;
    }
};

// Delete Strategy
async function deleteStrategy(id) {
    try {
        const result = await client.query('DELETE FROM strategies WHERE id = $1 RETURNING *;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error deleting strategy:', error);
        throw error;
    }
};

module.exports = {
    createStrategy,
    getAllStrategies,
    getStrategyById,
    getStrategyByName,
    updateStrategy,
    deleteStrategy
};