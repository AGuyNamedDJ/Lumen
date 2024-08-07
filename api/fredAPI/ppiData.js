const express = require('express');
require("dotenv").config();
const ppiRouter = express.Router();
const {
    storePPIData,
    getAllPPIData,
    getPPIDataByDate,
    updatePPIDataByDate,
    deletePPIDataByDate,
} = require('../../db/fredAPI/ppiData');
const axios = require('axios');

// Fetch and store PPI data from FRED API
const fetchPPIData = async () => {
    try {
        const response = await axios.get('https://api.stlouisfed.org/fred/series/observations', {
            params: {
                api_key: process.env.FRED_API_KEY,
                series_id: 'PPIACO',
                file_type: 'json'
            }
        });

        const data = response.data.observations;
        for (const entry of data) {
            const date = entry.date;
            const value = parseFloat(entry.value);
            await storePPIData({ date, value });
        }
    } catch (error) {
        console.error('Error fetching PPI data:', error);
    }
};

// Route to fetch PPI data from FRED and store in DB
ppiRouter.get('/fetch-ppi', async (req, res) => {
    try {
        await fetchPPIData();
        res.status(200).json({ message: 'PPI data fetched and stored successfully.' });
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch and store PPI data.' });
    }
});

// Create new PPI entry
ppiRouter.post('/ppi', async (req, res) => {
    const { date, value } = req.body;
    try {
        const newEntry = await storePPIData({ date, value });
        res.status(201).json(newEntry);
    } catch (error) {
        res.status(500).json({ error: 'Failed to create PPI data entry' });
    }
});

// Read all PPI entries
ppiRouter.get('/ppi', async (req, res) => {
    try {
        const result = await getAllPPIData();
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch PPI data' });
    }
});

// Read single PPI entry by date
ppiRouter.get('/ppi/:date', async (req, res) => {
    const { date } = req.params;
    try {
        const result = await getPPIDataByDate(date);
        if (!result) {
            return res.status(404).json({ error: 'PPI data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch PPI data' });
    }
});

// Update PPI entry by date
ppiRouter.put('/ppi/:date', async (req, res) => {
    const { date } = req.params;
    const { value } = req.body;
    try {
        const result = await updatePPIDataByDate(date, value);
        if (!result) {
            return res.status(404).json({ error: 'PPI data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to update PPI data' });
    }
});

// Delete PPI entry by date
ppiRouter.delete('/ppi/:date', async (req, res) => {
    const { date } = req.params;
    try {
        const result = await deletePPIDataByDate(date);
        if (!result) {
            return res.status(404).json({ error: 'PPI data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to delete PPI data' });
    }
});

module.exports = ppiRouter;