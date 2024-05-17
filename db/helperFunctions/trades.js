// Imports
const { client } = require("../index");

// Create Trade
async function createTrade({ strategy_id, open_time, close_time, status, entry_price, exit_price, profit_loss }) {
    try {
        const result = await client.query(`
            INSERT INTO trades (strategy_id, open_time, close_time, status, entry_price, exit_price, profit_loss)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING *;
        `, [strategy_id, open_time, close_time, status, entry_price, exit_price, profit_loss]);

        return result.rows[0];
    } catch (error) {
        console.error('Error creating trade:', error);
        throw error;
    }
};

// Get All Trades
async function getAllTrades() {
    try {
        const result = await client.query('SELECT * FROM trades;');
        return result.rows;
    } catch (error) {
        console.error('Error getting all trades:', error);
        throw error;
    }
};

// Get Trade By ID
async function getTradeById(id) {
    try {
        const result = await client.query('SELECT * FROM trades WHERE id = $1;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error getting trade by ID:', error);
        throw error;
    }
};

// Update Trade
async function updateTrade(id, fields = {}) {
    const setString = Object.keys(fields).map((key, index) => `"${key}"=$${index + 1}`).join(', ');

    try {
        const result = await client.query(`
            UPDATE trades
            SET ${setString}
            WHERE id = $${Object.keys(fields).length + 1}
            RETURNING *;
        `, [...Object.values(fields), id]);

        return result.rows[0];
    } catch (error) {
        console.error('Error updating trade:', error);
        throw error;
    }
};

// Delete Trade
async function deleteTrade(id) {
    try {
        const result = await client.query('DELETE FROM trades WHERE id = $1 RETURNING *;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error deleting trade:', error);
        throw error;
    }
};

module.exports = {
    createTrade,
    getAllTrades,
    getTradeById,
    updateTrade,
    deleteTrade
};





