const express = require('express');
require("dotenv").config();
const pceRouter = express.Router();
const {
    storePCEData,
    getAllPCEData,
    getPCEDataByDate,
    updatePCEDataByDate,
    deletePCEDataByDate,
} = require('../../db/fredAPI/pceData');
const axios = require('axios');

// Fetch and store PCE data from FRED API
const fetchPCEData = async () => {
    try {
        const response = await axios.get('https://api.stlouisfed.org/fred/series/observations', {
            params: {
                api_key: process.env.FRED_API_KEY,
                series_id: 'PCE',
                file_type: 'json'
            }
        });

        const data = response.data.observations;
        for (const entry of data) {
            const date = entry.date;
            const value = parseFloat(entry.value);

            await storePCEData({ date, value });
        }
    } catch (error) {
        console.error('Error fetching PCE data:', error);
    }
};

// Route to fetch PCE data from FRED and store in DB
pceRouter.get('/fetch-pce', async (req, res) => {
    try {
        await fetchPCEData();
        res.status(200).json({ message: 'PCE data fetched and stored successfully.' });
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch and store PCE data.' });
    }
});

// Create new PCE entry
pceRouter.post('/pce', async (req, res) => {
    const { date, value } = req.body;
    try {
        const newEntry = await storePCEData({ date, value });
        res.status(201).json(newEntry);
    } catch (error) {
        res.status(500).json({ error: 'Failed to create PCE data entry' });
    }
});

// Read all PCE entries
pceRouter.get('/pce', async (req, res) => {
    try {
        const result = await getAllPCEData();
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch PCE data' });
    }
});

// Read single PCE entry by date
pceRouter.get('/pce/:date', async (req, res) => {
    const { date } = req.params;
    try {
        const result = await getPCEDataByDate(date);
        if (!result) {
            return res.status(404).json({ error: 'PCE data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch PCE data' });
    }
});

// Update PCE entry by date
pceRouter.put('/pce/:date', async (req, res) => {
    const { date } = req.params;
    const { value } = req.body;
    try {
        const result = await updatePCEDataByDate(date, value);
        if (!result) {
            return res.status(404).json({ error: 'PCE data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to update PCE data' });
    }
});

// Delete PCE entry by date
pceRouter.delete('/pce/:date', async (req, res) => {
    const { date } = req.params;
    try {
        const result = await deletePCEDataByDate(date);
        if (!result) {
            return res.status(404).json({ error: 'PCE data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to delete PCE data' });
    }
});

module.exports = pceRouter;