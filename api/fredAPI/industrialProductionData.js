const express = require('express');
require("dotenv").config();
const industrialProductionRouter = express.Router();
const {
    storeIndustrialProductionData,
    getAllIndustrialProductionData,
    getIndustrialProductionDataByDate,
    updateIndustrialProductionDataByDate,
    deleteIndustrialProductionDataByDate,
} = require('../../db/fredAPI/industrialProductionData');
const axios = require('axios');

// Fetch and store Industrial Production data from FRED API
const fetchIndustrialProductionData = async () => {
    try {
        const response = await axios.get('https://api.stlouisfed.org/fred/series/observations', {
            params: {
                api_key: process.env.FRED_API_KEY,
                series_id: 'INDPRO',
                file_type: 'json'
            }
        });

        const data = response.data.observations;
        for (const entry of data) {
            const date = entry.date;
            const value = parseFloat(entry.value);

            await storeIndustrialProductionData({ date, value });
        }
    } catch (error) {
        console.error('Error fetching Industrial Production data:', error);
    }
};

// Route to fetch Industrial Production data from FRED and store in DB
industrialProductionRouter.get('/fetch-industrial-production', async (req, res) => {
    try {
        await fetchIndustrialProductionData();
        res.status(200).json({ message: 'Industrial Production data fetched and stored successfully.' });
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch and store Industrial Production data.' });
    }
});

// Create new Industrial Production entry
industrialProductionRouter.post('/industrial-production', async (req, res) => {
    const { date, value } = req.body;
    try {
        const newEntry = await storeIndustrialProductionData({ date, value });
        res.status(201).json(newEntry);
    } catch (error) {
        res.status(500).json({ error: 'Failed to create Industrial Production data entry' });
    }
});

// Read all Industrial Production entries
industrialProductionRouter.get('/industrial-production', async (req, res) => {
    try {
        const result = await getAllIndustrialProductionData();
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch Industrial Production data' });
    }
});

// Read single Industrial Production entry by date
industrialProductionRouter.get('/industrial-production/:date', async (req, res) => {
    const { date } = req.params;
    try {
        const result = await getIndustrialProductionDataByDate(date);
        if (!result) {
            return res.status(404).json({ error: 'Industrial Production data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch Industrial Production data' });
    }
});

// Update Industrial Production entry by date
industrialProductionRouter.put('/industrial-production/:date', async (req, res) => {
    const { date } = req.params;
    const { value } = req.body;
    try {
        const result = await updateIndustrialProductionDataByDate(date, value);
        if (!result) {
            return res.status(404).json({ error: 'Industrial Production data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to update Industrial Production data' });
    }
});

// Delete Industrial Production entry by date
industrialProductionRouter.delete('/industrial-production/:date', async (req, res) => {
    const { date } = req.params;
    try {
        const result = await deleteIndustrialProductionDataByDate(date);
        if (!result) {
            return res.status(404).json({ error: 'Industrial Production data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to delete Industrial Production data' });
    }
});

module.exports = industrialProductionRouter;