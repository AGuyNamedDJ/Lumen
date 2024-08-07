const express = require('express');
require("dotenv").config();
const cpiRouter = express.Router();
const {
    storeCPIData,
    getAllCPIData,
    getCPIDataByDate,
    updateCPIDataByDate,
    deleteCPIDataByDate,
} = require('../../db/fredAPI/cpiData');
const axios = require('axios');

// Fetch and store CPI data from FRED API
const fetchCPIData = async () => {
    try {
        const response = await axios.get('https://api.stlouisfed.org/fred/series/observations', {
            params: {
                api_key: process.env.FRED_API_KEY,
                series_id: 'CPIAUCSL',
                file_type: 'json'
            }
        });

        const data = response.data.observations;
        for (const entry of data) {
            const date = entry.date;
            const value = parseFloat(entry.value);
            await storeCPIData({ date, value });
        }
    } catch (error) {
        console.error('Error fetching CPI data:', error);
    }
};

// Route to fetch CPI data from FRED and store in DB
cpiRouter.get('/fetch-cpi', async (req, res) => {
    try {
        await fetchCPIData();
        res.status(200).json({ message: 'CPI data fetched and stored successfully.' });
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch and store CPI data.' });
    }
});

// Create new CPI entry
cpiRouter.post('/cpi', async (req, res) => {
    const { date, value } = req.body;
    try {
        const newEntry = await storeCPIData({ date, value });
        res.status(201).json(newEntry);
    } catch (error) {
        res.status(500).json({ error: 'Failed to create CPI data entry' });
    }
});

// Read all CPI entries
cpiRouter.get('/', async (req, res) => {
    try {
        const result = await getAllCPIData();
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch CPI data' });
    }
});

// Read single CPI entry by date
cpiRouter.get('/:date', async (req, res) => {
    const { date } = req.params;
    try {
        const result = await getCPIDataByDate(date);
        if (!result) {
            return res.status(404).json({ error: 'CPI data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch CPI data' });
    }
});

// Update CPI entry by date
cpiRouter.put('/:date', async (req, res) => {
    const { date } = req.params;
    const { value } = req.body;
    try {
        const result = await updateCPIDataByDate(date, value);
        if (!result) {
            return res.status(404).json({ error: 'CPI data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to update CPI data' });
    }
});

// Delete CPI entry by date
cpiRouter.delete('/:date', async (req, res) => {
    const { date } = req.params;
    try {
        const result = await deleteCPIDataByDate(date);
        if (!result) {
            return res.status(404).json({ error: 'CPI data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to delete CPI data' });
    }
});

module.exports = cpiRouter;