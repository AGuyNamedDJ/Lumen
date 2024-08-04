const { client } = require("../index");

// Convert date string to Unix timestamp
function convertToTimestamp(dateString) {
    return new Date(dateString).getTime();
}

// Store SMA data in the database
async function storeSMAData(smaData) {
    const queryText = `
        INSERT INTO sma_indicators (symbol, period, timespan, timestamp, value)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING *;
    `;

    try {
        await client.query('BEGIN');

        for (const data of smaData) {
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
        console.error('Error storing SMA data:', error);
        throw error;
    }
}

// Get all SMA indicators
async function getAllSMAIndicators() {
    try {
        const result = await client.query('SELECT * FROM sma_indicators;');
        return result.rows;
    } catch (error) {
        console.error('Error getting all SMA indicators:', error);
        throw error;
    }
}

// Get SMA indicator by ID
async function getSMAIndicatorById(id) {
    try {
        const result = await client.query('SELECT * FROM sma_indicators WHERE id = $1;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error getting SMA indicator by ID:', error);
        throw error;
    }
}

// Get SMA indicators by symbol
async function getSMAIndicatorsBySymbol(symbol) {
    try {
        const result = await client.query('SELECT * FROM sma_indicators WHERE symbol = $1;', [symbol]);
        return result.rows;
    } catch (error) {
        console.error('Error getting SMA indicators by symbol:', error);
        throw error;
    }
}

// Delete SMA indicator
async function deleteSMAIndicator(id) {
    try {
        const result = await client.query('DELETE FROM sma_indicators WHERE id = $1 RETURNING *;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error deleting SMA indicator:', error);
        throw error;
    }
}

module.exports = {
    storeSMAData,
    getAllSMAIndicators,
    getSMAIndicatorById,
    getSMAIndicatorsBySymbol,
    deleteSMAIndicator
};