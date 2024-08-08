const { client } = require('../index');

// Store Consumer Confidence data
const storeConsumerConfidenceData = async ({ date, value }) => {
    try {
        console.log(`Storing Consumer Confidence Data: Date - ${date}, Value - ${value}`);
        const query = `
            INSERT INTO consumer_confidence_data (date, value)
            VALUES ($1, $2)
            ON CONFLICT (date) DO NOTHING
            RETURNING *;
        `;
        const values = [date, value];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error storing Consumer Confidence data:', error);
    }
};

// Get all Consumer Confidence data
const getAllConsumerConfidenceData = async () => {
    try {
        console.log('Fetching all Consumer Confidence data');
        const query = 'SELECT * FROM consumer_confidence_data ORDER BY date DESC;';
        const res = await client.query(query);
        return res.rows;
    } catch (error) {
        console.error('Error fetching Consumer Confidence data:', error);
    }
};

// Get Consumer Confidence data by date
const getConsumerConfidenceDataByDate = async (date) => {
    try {
        console.log(`Fetching Consumer Confidence data for date: ${date}`);
        const query = 'SELECT * FROM consumer_confidence_data WHERE date = $1;';
        const values = [date];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error fetching Consumer Confidence data by date:', error);
    }
};

// Update Consumer Confidence data by date
const updateConsumerConfidenceDataByDate = async (date, value) => {
    try {
        console.log(`Updating Consumer Confidence data for date: ${date} with value: ${value}`);
        const query = `
            UPDATE consumer_confidence_data
            SET value = $2, updated_at = CURRENT_TIMESTAMP
            WHERE date = $1
            RETURNING *;
        `;
        const values = [date, value];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error updating Consumer Confidence data by date:', error);
    }
};

// Delete Consumer Confidence data by date
const deleteConsumerConfidenceDataByDate = async (date) => {
    try {
        console.log(`Deleting Consumer Confidence data for date: ${date}`);
        const query = 'DELETE FROM consumer_confidence_data WHERE date = $1 RETURNING *;';
        const values = [date];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error deleting Consumer Confidence data by date:', error);
    }
};

module.exports = {
    storeConsumerConfidenceData,
    getAllConsumerConfidenceData,
    getConsumerConfidenceDataByDate,
    updateConsumerConfidenceDataByDate,
    deleteConsumerConfidenceDataByDate,
};