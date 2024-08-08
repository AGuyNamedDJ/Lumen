const { client } = require('../index');

// Store Consumer Sentiment data
const storeConsumerSentimentData = async ({ date, value }) => {
    try {
        console.log(`Storing Consumer Sentiment Data: Date - ${date}, Value - ${value}`);
        const query = `
            INSERT INTO consumer_sentiment_data (date, value)
            VALUES ($1, $2)
            ON CONFLICT (date) DO NOTHING
            RETURNING *;
        `;
        const values = [date, value];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error storing Consumer Sentiment data:', error);
    }
};

// Get all Consumer Sentiment data
const getAllConsumerSentimentData = async () => {
    try {
        console.log('Fetching all Consumer Sentiment data');
        const query = 'SELECT * FROM consumer_sentiment_data ORDER BY date DESC;';
        const res = await client.query(query);
        return res.rows;
    } catch (error) {
        console.error('Error fetching Consumer Sentiment data:', error);
    }
};

// Get Consumer Sentiment data by date
const getConsumerSentimentDataByDate = async (date) => {
    try {
        console.log(`Fetching Consumer Sentiment data for date: ${date}`);
        const query = 'SELECT * FROM consumer_sentiment_data WHERE date = $1;';
        const values = [date];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error fetching Consumer Sentiment data by date:', error);
    }
};

// Update Consumer Sentiment data by date
const updateConsumerSentimentDataByDate = async (date, value) => {
    try {
        console.log(`Updating Consumer Sentiment data for date: ${date} with value: ${value}`);
        const query = `
            UPDATE consumer_sentiment_data
            SET value = $2, updated_at = CURRENT_TIMESTAMP
            WHERE date = $1
            RETURNING *;
        `;
        const values = [date, value];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error updating Consumer Sentiment data by date:', error);
    }
};

// Delete Consumer Sentiment data by date
const deleteConsumerSentimentDataByDate = async (date) => {
    try {
        console.log(`Deleting Consumer Sentiment data for date: ${date}`);
        const query = 'DELETE FROM consumer_sentiment_data WHERE date = $1 RETURNING *;';
        const values = [date];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error deleting Consumer Sentiment data by date:', error);
    }
};

module.exports = {
    storeConsumerSentimentData,
    getAllConsumerSentimentData,
    getConsumerSentimentDataByDate,
    updateConsumerSentimentDataByDate,
    deleteConsumerSentimentDataByDate,
};