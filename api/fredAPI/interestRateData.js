const express = require('express');
require("dotenv").config();
const interestRateRouter = express.Router();
const {
    storeInterestRateData,
    getAllInterestRateData,
    getInterestRateDataByDate,
    updateInterestRateDataByDate,
    deleteInterestRateDataByDate,
} = require('../../db/fredAPI/interestRateData');
const axios = require('axios');

// Series IDs for different interest rates
const seriesIDs = ['FEDFUNDS', 'DPRIME', 'GS10', 'TB3MS', 'GS1', 'MORTGAGE30US', 'GS30'];

// Fetch and store Interest Rate data from FRED API
const fetchInterestRateData = async (series_id) => {
    try {
        const response = await axios.get('https://api.stlouisfed.org/fred/series/observations', {
            params: {
                api_key: process.env.FRED_API_KEY,
                series_id: series_id,
                file_type: 'json'
            }
        });

        const data = response.data.observations;
        for (const entry of data) {
            const date = entry.date;
            const value = parseFloat(entry.value);

            await storeInterestRateData({ date, series_id, value });
        }
    } catch (error) {
        console.error(`Error fetching Interest Rate data for ${series_id}:`, error);
    }
};

// Route to fetch Interest Rate data from FRED and store in DB
interestRateRouter.get('/fetch-interest-rates', async (req, res) => {
    try {
        for (const series_id of seriesIDs) {
            await fetchInterestRateData(series_id);
        }
        res.status(200).json({ message: 'Interest Rate data fetched and stored successfully.' });
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch and store Interest Rate data.' });
    }
});

// Create new Interest Rate entry
interestRateRouter.post('/interest-rate', async (req, res) => {
    const { date, series_id, value } = req.body;
    try {
        const newEntry = await storeInterestRateData({ date, series_id, value });
        res.status(201).json(newEntry);
    } catch (error) {
        res.status(500).json({ error: 'Failed to create Interest Rate data entry' });
    }
});

// Read all Interest Rate entries
interestRateRouter.get('/interest-rate', async (req, res) => {
    try {
        const result = await getAllInterestRateData();
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch Interest Rate data' });
    }
});

// Read single Interest Rate entry by date
interestRateRouter.get('/interest-rate/:date', async (req, res) => {
    const { date } = req.params;
    try {
        const result = await getInterestRateDataByDate(date);
        if (!result) {
            return res.status(404).json({ error: 'Interest Rate data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch Interest Rate data' });
    }
});

// Update Interest Rate entry by date
interestRateRouter.put('/interest-rate/:date', async (req, res) => {
    const { date } = req.params;
    const { series_id, value } = req.body;
    try {
        const result = await updateInterestRateDataByDate(date, series_id, value);
        if (!result) {
            return res.status(404).json({ error: 'Interest Rate data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to update Interest Rate data' });
    }
});

// Delete Interest Rate entry by date
interestRateRouter.delete('/interest-rate/:date', async (req, res) => {
    const { date } = req.params;
    const { series_id } = req.body;
    try {
        const result = await deleteInterestRateDataByDate(date, series_id);
        if (!result) {
            return res.status(404).json({ error: 'Interest Rate data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to delete Interest Rate data' });
    }
});

module.exports = interestRateRouter;