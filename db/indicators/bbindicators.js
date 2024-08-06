const { client } = require("../index");

// Store Bollinger Bands Data
async function storeBBData(data) {
    try {
        await client.query('BEGIN');
        for (const record of data) {
            await client.query(`
                INSERT INTO bb_indicators (symbol, period, timespan, timestamp, middle_band, upper_band, lower_band)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (symbol, period, timespan, timestamp) DO UPDATE
                SET middle_band = EXCLUDED.middle_band,
                    upper_band = EXCLUDED.upper_band,
                    lower_band = EXCLUDED.lower_band,
                    updated_at = CURRENT_TIMESTAMP;
            `, [record.symbol, record.period, record.timespan, record.timestamp, record.middle_band, record.upper_band, record.lower_band]);
        }
        await client.query('COMMIT');
    } catch (error) {
        await client.query('ROLLBACK');
        console.error('Error storing BB data:', error);
        throw error;
    }
}

// Get All Bollinger Bands Indicators
async function getAllBBIndicators() {
    try {
        const result = await client.query('SELECT * FROM bb_indicators;');
        return result.rows;
    } catch (error) {
        console.error('Error getting all BB indicators:', error);
        throw error;
    }
}

// Get Bollinger Bands Indicator By ID
async function getBBIndicatorById(id) {
    try {
        const result = await client.query('SELECT * FROM bb_indicators WHERE id = $1;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error getting BB indicator by ID:', error);
        throw error;
    }
}

// Get Bollinger Bands Indicators By Symbol
async function getBBIndicatorsBySymbol(symbol) {
    try {
        const result = await client.query('SELECT * FROM bb_indicators WHERE symbol = $1;', [symbol]);
        return result.rows;
    } catch (error) {
        console.error('Error getting BB indicators by symbol:', error);
        throw error;
    }
}

// Delete Bollinger Bands Indicator
async function deleteBBIndicator(id) {
    try {
        const result = await client.query('DELETE FROM bb_indicators WHERE id = $1 RETURNING *;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error deleting BB indicator:', error);
        throw error;
    }
}

module.exports = {
    storeBBData,
    getAllBBIndicators,
    getBBIndicatorById,
    getBBIndicatorsBySymbol,
    deleteBBIndicator
};