const express = require('express');
require("dotenv").config();
const consumerConfidenceRouter = express.Router();
const {
    storeConsumerConfidenceData,
    getAllConsumerConfidenceData,
    getConsumerConfidenceDataByDate,
    updateConsumerConfidenceDataByDate,
    deleteConsumerConfidenceDataByDate,
} = require('../../db/fredAPI/consumerConfidenceData');
const axios = require('axios');

// Fetch and store Consumer Confidence data from FRED API
const fetchConsumerConfidenceData = async () => {
    try {
        const response = await axios.get('https://api.stlouisfed.org/fred/series/observations', {
            params: {
                api_key: process.env.FRED_API_KEY,
                series_id: 'CSCICP03USM665S',
                file_type: 'json'
            }
        });

        const data = response.data.observations;
        for (const entry of data) {
            const date = entry.date;
            const value = parseFloat(entry.value);

            await storeConsumerConfidenceData({ date, value });
        }
    } catch (error) {
        console.error('Error fetching Consumer Confidence data:', error);
    }
};

// Route to fetch Consumer Confidence data from FRED and store in DB
consumerConfidenceRouter.get('/fetch-consumer-confidence', async (req, res) => {
    try {
        await fetchConsumerConfidenceData();
        res.status(200).json({ message: 'Consumer Confidence data fetched and stored successfully.' });
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch and store Consumer Confidence data.' });
    }
});

// Create new Consumer Confidence entry
consumerConfidenceRouter.post('/consumer-confidence', async (req, res) => {
    const { date, value } = req.body;
    try {
        const newEntry = await storeConsumerConfidenceData({ date, value });
        res.status(201).json(newEntry);
    } catch (error) {
        res.status(500).json({ error: 'Failed to create Consumer Confidence data entry' });
    }
});

// Read all Consumer Confidence entries
consumerConfidenceRouter.get('/consumer-confidence', async (req, res) => {
    try {
        const result = await getAllConsumerConfidenceData();
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch Consumer Confidence data' });
    }
});

// Read single Consumer Confidence entry by date
consumerConfidenceRouter.get('/consumer-confidence/:date', async (req, res) => {
    const { date } = req.params;
    try {
        const result = await getConsumerConfidenceDataByDate(date);
        if (!result) {
            return res.status(404).json({ error: 'Consumer Confidence data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch Consumer Confidence data' });
    }
});

// Update Consumer Confidence entry by date
consumerConfidenceRouter.put('/consumer-confidence/:date', async (req, res) => {
    const { date } = req.params;
    const { value } = req.body;
    try {
        const result = await updateConsumerConfidenceDataByDate(date, value);
        if (!result) {
            return res.status(404).json({ error: 'Consumer Confidence data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to update Consumer Confidence data' });
    }
});

// Delete Consumer Confidence entry by date
consumerConfidenceRouter.delete('/consumer-confidence/:date', async (req, res) => {
    const { date } = req.params;
    try {
        const result = await deleteConsumerConfidenceDataByDate(date);
        if (!result) {
            return res.status(404).json({ error: 'Consumer Confidence data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to delete Consumer Confidence data' });
    }
});

module.exports = consumerConfidenceRouter;