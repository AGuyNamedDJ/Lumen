// Imports
const { client } = require("../index");

// Store RSI Data
async function storeRSIData(data) {
    try {
        const values = data.map(({ symbol, period, timespan, timestamp, value }) => 
            `('${symbol}', ${period}, '${timespan}', ${timestamp}, ${value})`).join(',');

        const query = `
            INSERT INTO rsi_indicators (symbol, period, timespan, timestamp, value)
            VALUES ${values}
            ON CONFLICT (symbol, period, timespan, timestamp) DO NOTHING;
        `;

        await client.query(query);
        console.log('RSI data stored successfully');
    } catch (error) {
        console.error('Error storing RSI data:', error);
        throw error;
    }
}

// Get All RSI Indicators
async function getAllRSIIndicators() {
    try {
        const result = await client.query('SELECT * FROM rsi_indicators;');
        return result.rows;
    } catch (error) {
        console.error('Error getting all RSI indicators:', error);
        throw error;
    }
}

// Get RSI Indicator By ID
async function getRSIIndicatorById(id) {
    try {
        const result = await client.query('SELECT * FROM rsi_indicators WHERE id = $1;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error getting RSI indicator by ID:', error);
        throw error;
    }
}

// Get RSI Indicators By Symbol
async function getRSIIndicatorsBySymbol(symbol) {
    try {
        const result = await client.query('SELECT * FROM rsi_indicators WHERE symbol = $1;', [symbol]);
        return result.rows;
    } catch (error) {
        console.error('Error getting RSI indicators by symbol:', error);
        throw error;
    }
}

// Delete RSI Indicator
async function deleteRSIIndicator(id) {
    try {
        const result = await client.query('DELETE FROM rsi_indicators WHERE id = $1 RETURNING *;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error deleting RSI indicator:', error);
        throw error;
    }
};

module.exports = {
    storeRSIData,
    getAllRSIIndicators,
    getRSIIndicatorById,
    getRSIIndicatorsBySymbol,
    deleteRSIIndicator
};