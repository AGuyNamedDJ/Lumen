// Imports
const { client } = require("../index");

// Create MACD Indicator
async function storeMACDData(data) {
    try {
        await client.query('BEGIN');
        const insertPromises = data.map(record => {
            const { symbol, period, timespan, timestamp, macd_line, signal_line, histogram } = record;
            return client.query(`
                INSERT INTO macd_indicators (symbol, period, timespan, timestamp, macd_line, signal_line, histogram)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (symbol, period, timespan, timestamp)
                DO NOTHING;
            `, [symbol, period, timespan, timestamp, macd_line, signal_line, histogram]);
        });

        await Promise.all(insertPromises);
        await client.query('COMMIT');
    } catch (error) {
        await client.query('ROLLBACK');
        console.error('Error storing MACD data:', error);
        throw error;
    }
};

// Get All MACD Indicators
async function getAllMACDIndicators() {
    try {
        const result = await client.query('SELECT * FROM macd_indicators;');
        return result.rows;
    } catch (error) {
        console.error('Error getting all MACD indicators:', error);
        throw error;
    }
};

// Get MACD Indicator By ID
async function getMACDIndicatorById(id) {
    try {
        const result = await client.query('SELECT * FROM macd_indicators WHERE id = $1;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error getting MACD indicator by ID:', error);
        throw error;
    }
};

// Get MACD Indicators By Symbol
async function getMACDIndicatorsBySymbol(symbol) {
    try {
        const result = await client.query('SELECT * FROM macd_indicators WHERE symbol = $1;', [symbol]);
        return result.rows;
    } catch (error) {
        console.error('Error getting MACD indicators by symbol:', error);
        throw error;
    }
};

// Delete MACD Indicator
async function deleteMACDIndicator(id) {
    try {
        const result = await client.query('DELETE FROM macd_indicators WHERE id = $1 RETURNING *;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error deleting MACD indicator:', error);
        throw error;
    }
};

module.exports = {
    storeMACDData,
    getAllMACDIndicators,
    getMACDIndicatorById,
    getMACDIndicatorsBySymbol,
    deleteMACDIndicator
};