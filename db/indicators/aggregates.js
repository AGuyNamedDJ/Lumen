// db/indicators/aggregates.js

const { client } = require("../index");

// Store Aggregates Data
async function storeAggregatesData(aggregatesData) {
    const query = `
        INSERT INTO aggregates (symbol, multiplier, timespan, timestamp, open, high, low, close, volume)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT (symbol, multiplier, timespan, timestamp)
        DO UPDATE SET open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low, close = EXCLUDED.close, volume = EXCLUDED.volume, updated_at = CURRENT_TIMESTAMP
        RETURNING *;
    `;

    try {
        await client.query('BEGIN');
        const results = [];
        for (const data of aggregatesData) {
            const result = await client.query(query, [
                data.symbol,
                data.multiplier,
                data.timespan,
                data.timestamp,
                data.open,
                data.high,
                data.low,
                data.close,
                data.volume
            ]);
            results.push(result.rows[0]);
        }
        await client.query('COMMIT');
        return results;
    } catch (error) {
        await client.query('ROLLBACK');
        console.error('Error storing aggregates data:', error);
        throw error;
    }
};

// Get All Aggregates
async function getAllAggregates() {
    try {
        const result = await client.query('SELECT * FROM aggregates;');
        return result.rows;
    } catch (error) {
        console.error('Error getting all aggregates:', error);
        throw error;
    }
};

// Get Aggregates By Symbol
async function getAggregatesBySymbol(symbol) {
    try {
        const result = await client.query('SELECT * FROM aggregates WHERE symbol = $1;', [symbol]);
        return result.rows;
    } catch (error) {
        console.error('Error getting aggregates by symbol:', error);
        throw error;
    }
};

// Get Aggregates By Symbol and Timespan
async function getAggregatesBySymbolAndTimespan(symbol, timespan) {
    try {
        const result = await client.query('SELECT * FROM aggregates WHERE symbol = $1 AND timespan = $2;', [symbol, timespan]);
        return result.rows;
    } catch (error) {
        console.error('Error getting aggregates by symbol and timespan:', error);
        throw error;
    }
};

// Delete Aggregates
async function deleteAggregates(id) {
    try {
        const result = await client.query('DELETE FROM aggregates WHERE id = $1 RETURNING *;', [id]);
        return result.rows[0];
    } catch (error) {
        console.error('Error deleting aggregates:', error);
        throw error;
    }
};

module.exports = {
    storeAggregatesData,
    getAllAggregates,
    getAggregatesBySymbol,
    getAggregatesBySymbolAndTimespan,
    deleteAggregates
};