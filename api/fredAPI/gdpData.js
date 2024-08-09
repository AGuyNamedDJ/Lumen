const express = require('express');
require("dotenv").config();
const gdpRouter = express.Router();
const {
    storeGDPData,
    getAllGDPData,
    getGDPDataByDate,
    updateGDPDataByDate,
    deleteGDPDataByDate,
} = require('../../db/fredAPI/gdpData');
const axios = require('axios');

// Route to fetch GDP data from FRED and store in DB
gdpRouter.get('/fetch-gdp', async (req, res) => {
    try {
        await fetchGDPData();
        res.status(200).json({ message: 'GDP data fetched and stored successfully.' });
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch and store GDP data.' });
    }
});

// Create new GDP entry
gdpRouter.post('/gdp', async (req, res) => {
    const { date, value } = req.body;
    try {
        const newEntry = await storeGDPData({ date, value });
        res.status(201).json(newEntry);
    } catch (error) {
        res.status(500).json({ error: 'Failed to create GDP data entry' });
    }
});

// Read all GDP entries
gdpRouter.get('/gdp', async (req, res) => {
    try {
        const result = await getAllGDPData();
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch GDP data' });
    }
});

// Read single GDP entry by date
gdpRouter.get('/gdp/:date', async (req, res) => {
    const { date } = req.params;
    try {
        const result = await getGDPDataByDate(date);
        if (!result) {
            return res.status(404).json({ error: 'GDP data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch GDP data' });
    }
});

// Update GDP entry by date
gdpRouter.put('/gdp/:date', async (req, res) => {
    const { date } = req.params;
    const { value } = req.body;
    try {
        const result = await updateGDPDataByDate(date, value);
        if (!result) {
            return res.status(404).json({ error: 'GDP data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to update GDP data' });
    }
});

// Delete GDP entry by date
gdpRouter.delete('/gdp/:date', async (req, res) => {
    const { date } = req.params;
    try {
        const result = await deleteGDPDataByDate(date);
        if (!result) {
            return res.status(404).json({ error: 'GDP data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to delete GDP data' });
    }
});

module.exports = gdpRouter;