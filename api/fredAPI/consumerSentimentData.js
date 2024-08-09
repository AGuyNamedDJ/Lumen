const express = require('express');
require("dotenv").config();
const consumerSentimentRouter = express.Router();
const {
    storeConsumerSentimentData,
    getAllConsumerSentimentData,
    getConsumerSentimentDataByDate,
    updateConsumerSentimentDataByDate,
    deleteConsumerSentimentDataByDate,
} = require('../../db/fredAPI/consumerSentimentData');
const axios = require('axios');

// Route to fetch Consumer Sentiment data from FRED and store in DB
consumerSentimentRouter.get('/fetch-consumer-sentiment', async (req, res) => {
    try {
        await fetchConsumerSentimentData();
        res.status(200).json({ message: 'Consumer Sentiment data fetched and stored successfully.' });
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch and store Consumer Sentiment data.' });
    }
});

// Create new Consumer Sentiment entry
consumerSentimentRouter.post('/consumer-sentiment', async (req, res) => {
    const { date, value } = req.body;
    try {
        const newEntry = await storeConsumerSentimentData({ date, value });
        res.status(201).json(newEntry);
    } catch (error) {
        res.status(500).json({ error: 'Failed to create Consumer Sentiment data entry' });
    }
});

// Read all Consumer Sentiment entries
consumerSentimentRouter.get('/consumer-sentiment', async (req, res) => {
    try {
        const result = await getAllConsumerSentimentData();
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch Consumer Sentiment data' });
    }
});

// Read single Consumer Sentiment entry by date
consumerSentimentRouter.get('/consumer-sentiment/:date', async (req, res) => {
    const { date } = req.params;
    try {
        const result = await getConsumerSentimentDataByDate(date);
        if (!result) {
            return res.status(404).json({ error: 'Consumer Sentiment data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch Consumer Sentiment data' });
    }
});

// Update Consumer Sentiment entry by date
consumerSentimentRouter.put('/consumer-sentiment/:date', async (req, res) => {
    const { date } = req.params;
    const { value } = req.body;
    try {
        const result = await updateConsumerSentimentDataByDate(date, value);
        if (!result) {
            return res.status(404).json({ error: 'Consumer Sentiment data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to update Consumer Sentiment data' });
    }
});

// Delete Consumer Sentiment entry by date
consumerSentimentRouter.delete('/consumer-sentiment/:date', async (req, res) => {
    const { date } = req.params;
    try {
        const result = await deleteConsumerSentimentDataByDate(date);
        if (!result) {
            return res.status(404).json({ error: 'Consumer Sentiment data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to delete Consumer Sentiment data' });
    }
});

module.exports = consumerSentimentRouter;