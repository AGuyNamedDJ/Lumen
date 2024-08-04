const { client } = require("../index");

// Convert date string to Unix timestamp
function convertToTimestamp(dateString) {
    return new Date(dateString).getTime();
}

// Store SMA data in the database
async function storeEMAData(emaData) {
    const queryText = `
        INSERT INTO ema_indicators (symbol, period, timespan, timestamp, value)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING *;
    `;

    try {
        await client.query('BEGIN');

        for (const data of emaData) {
            const timestamp = convertToTimestamp(data.timestamp);
            await client.query(queryText, [
                data.symbol,
                data.period,
                data.timespan,
                timestamp,
                data.value
            ]);
        }

        await client.query('COMMIT');
    } catch (error) {
        await client.query('ROLLBACK');
        console.error('Error storing EMA data:', error);
        throw error;
    }
}

// Get all EMA indicators
async function getAllEMAIndicators() {
    try {
        const result = await client.query('SELECT * FROM ema_indicators;');
        return result.rows;
    } catch (error) {
        console.error('Error getting all EMA indicators:', error);
        throw error;
    }
}

// Get EMA indicator by ID
async function getEMAIndicatorById(id) {
    try {
        const result = await client.query('SELECT * FROM ema_indicators WHERE id = $1;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error getting EMA indicator by ID:', error);
        throw error;
    }
}

// Get EMA indicators by symbol
async function getEMAIndicatorsBySymbol(symbol) {
    try {
        const result = await client.query('SELECT * FROM ema_indicators WHERE symbol = $1;', [symbol]);
        return result.rows;
    } catch (error) {
        console.error('Error getting EMA indicators by symbol:', error);
        throw error;
    }
}

// Delete EMA indicator
async function deleteEMAIndicator(id) {
    try {
        const result = await client.query('DELETE FROM ema_indicators WHERE id = $1 RETURNING *;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error deleting EMA indicator:', error);
        throw error;
    }
}

module.exports = {
    storeEMAData,
    getAllEMAIndicators,
    getEMAIndicatorById,
    getEMAIndicatorsBySymbol,
    deleteEMAIndicator
};