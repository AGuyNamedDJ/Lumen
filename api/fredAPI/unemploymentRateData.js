const express = require('express');
require("dotenv").config();
const unemploymentRateRouter = express.Router();
const {
    storeUnemploymentRateData,
    getAllUnemploymentRateData,
    getUnemploymentRateDataByDate,
    updateUnemploymentRateDataByDate,
    deleteUnemploymentRateDataByDate,
} = require('../../db/fredAPI/unemploymentRateData');
const axios = require('axios');

// Fetch and store Unemployment Rate data from FRED API
const fetchUnemploymentRateData = async () => {
    try {
        const response = await axios.get('https://api.stlouisfed.org/fred/series/observations', {
            params: {
                api_key: process.env.FRED_API_KEY,
                series_id: 'UNRATE',
                file_type: 'json'
            }
        });

        const data = response.data.observations;
        for (const entry of data) {
            const date = entry.date;
            const value = parseFloat(entry.value);

            await storeUnemploymentRateData({ date, value });
        }
    } catch (error) {
        console.error('Error fetching Unemployment Rate data:', error);
    }
};

// Route to fetch Unemployment Rate data from FRED and store in DB
unemploymentRateRouter.get('/fetch-unemployment-rate', async (req, res) => {
    try {
        await fetchUnemploymentRateData();
        res.status(200).json({ message: 'Unemployment Rate data fetched and stored successfully.' });
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch and store Unemployment Rate data.' });
    }
});

// Create new Unemployment Rate entry
unemploymentRateRouter.post('/unemployment-rate', async (req, res) => {
    const { date, value } = req.body;
    try {
        const newEntry = await storeUnemploymentRateData({ date, value });
        res.status(201).json(newEntry);
    } catch (error) {
        res.status(500).json({ error: 'Failed to create Unemployment Rate data entry' });
    }
});

// Read all Unemployment Rate entries
unemploymentRateRouter.get('/unemployment-rate', async (req, res) => {
    try {
        const result = await getAllUnemploymentRateData();
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch Unemployment Rate data' });
    }
});

// Read single Unemployment Rate entry by date
unemploymentRateRouter.get('/unemployment-rate/:date', async (req, res) => {
    const { date } = req.params;
    try {
        const result = await getUnemploymentRateDataByDate(date);
        if (!result) {
            return res.status(404).json({ error: 'Unemployment Rate data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch Unemployment Rate data' });
    }
});

// Update Unemployment Rate entry by date
unemploymentRateRouter.put('/unemployment-rate/:date', async (req, res) => {
    const { date } = req.params;
    const { value } = req.body;
    try {
        const result = await updateUnemploymentRateDataByDate(date, value);
        if (!result) {
            return res.status(404).json({ error: 'Unemployment Rate data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to update Unemployment Rate data' });
    }
});

// Delete Unemployment Rate entry by date
unemploymentRateRouter.delete('/unemployment-rate/:date', async (req, res) => {
    const { date } = req.params;
    try {
        const result = await deleteUnemploymentRateDataByDate(date);
        if (!result) {
            return res.status(404).json({ error: 'Unemployment Rate data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to delete Unemployment Rate data' });
    }
});

module.exports = unemploymentRateRouter;