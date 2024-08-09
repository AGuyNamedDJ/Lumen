const express = require('express');
require("dotenv").config();
const coreInflationRouter = express.Router();
const {
    storeCoreInflationData,
    getAllCoreInflationData,
    getCoreInflationDataByDate,
    updateCoreInflationDataByDate,
    deleteCoreInflationDataByDate,
} = require('../../db/fredAPI/coreInflationData');
const axios = require('axios');

// Route to fetch Core Inflation data from FRED and store in DB
coreInflationRouter.get('/fetch-core-inflation', async (req, res) => {
    try {
        await fetchCoreInflationData();
        res.status(200).json({ message: 'Core Inflation data fetched and stored successfully.' });
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch and store Core Inflation data.' });
    }
});

// Create new Core Inflation entry
coreInflationRouter.post('/core-inflation', async (req, res) => {
    const { date, value } = req.body;
    try {
        const newEntry = await storeCoreInflationData({ date, value });
        res.status(201).json(newEntry);
    } catch (error) {
        res.status(500).json({ error: 'Failed to create Core Inflation data entry' });
    }
});

// Read all Core Inflation entries
coreInflationRouter.get('/core-inflation', async (req, res) => {
    try {
        const result = await getAllCoreInflationData();
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch Core Inflation data' });
    }
});

// Read single Core Inflation entry by date
coreInflationRouter.get('/core-inflation/:date', async (req, res) => {
    const { date } = req.params;
    try {
        const result = await getCoreInflationDataByDate(date);
        if (!result) {
            return res.status(404).json({ error: 'Core Inflation data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch Core Inflation data' });
    }
});

// Update Core Inflation entry by date
coreInflationRouter.put('/core-inflation/:date', async (req, res) => {
    const { date } = req.params;
    const { value } = req.body;
    try {
        const result = await updateCoreInflationDataByDate(date, value);
        if (!result) {
            return res.status(404).json({ error: 'Core Inflation data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to update Core Inflation data' });
    }
});

// Delete Core Inflation entry by date
coreInflationRouter.delete('/core-inflation/:date', async (req, res) => {
    const { date } = req.params;
    try {
        const result = await deleteCoreInflationDataByDate(date);
        if (!result) {
            return res.status(404).json({ error: 'Core Inflation data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to delete Core Inflation data' });
    }
});

module.exports = coreInflationRouter;