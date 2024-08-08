const express = require('express');
require("dotenv").config();
const laborForceParticipationRouter = express.Router();
const {
    storeLaborForceParticipationData,
    getAllLaborForceParticipationData,
    getLaborForceParticipationDataByDate,
    updateLaborForceParticipationDataByDate,
    deleteLaborForceParticipationDataByDate,
} = require('../../db/fredAPI/laborForceParticipationRateData');
const axios = require('axios');

// Series ID for Labor Force Participation Rate
const seriesID = 'CIVPART';

// Fetch and store Labor Force Participation data from FRED API
const fetchLaborForceParticipationData = async () => {
    try {
        const response = await axios.get('https://api.stlouisfed.org/fred/series/observations', {
            params: {
                api_key: process.env.FRED_API_KEY,
                series_id: seriesID,
                file_type: 'json'
            }
        });

        const data = response.data.observations;
        for (const entry of data) {
            const date = entry.date;
            const value = parseFloat(entry.value);

            await storeLaborForceParticipationData({ date, value });
        }
    } catch (error) {
        console.error('Error fetching Labor Force Participation data:', error);
    }
};

// Route to fetch Labor Force Participation data from FRED and store in DB
laborForceParticipationRouter.get('/fetch-labor-force-participation', async (req, res) => {
    try {
        await fetchLaborForceParticipationData();
        res.status(200).json({ message: 'Labor Force Participation data fetched and stored successfully.' });
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch and store Labor Force Participation data.' });
    }
});

// Create new Labor Force Participation entry
laborForceParticipationRouter.post('/labor-force-participation', async (req, res) => {
    const { date, value } = req.body;
    try {
        const newEntry = await storeLaborForceParticipationData({ date, value });
        res.status(201).json(newEntry);
    } catch (error) {
        res.status(500).json({ error: 'Failed to create Labor Force Participation data entry' });
    }
});

// Read all Labor Force Participation entries
laborForceParticipationRouter.get('/labor-force-participation', async (req, res) => {
    try {
        const result = await getAllLaborForceParticipationData();
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch Labor Force Participation data' });
    }
});

// Read single Labor Force Participation entry by date
laborForceParticipationRouter.get('/labor-force-participation/:date', async (req, res) => {
    const { date } = req.params;
    try {
        const result = await getLaborForceParticipationDataByDate(date);
        if (!result) {
            return res.status(404).json({ error: 'Labor Force Participation data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch Labor Force Participation data' });
    }
});

// Update Labor Force Participation entry by date
laborForceParticipationRouter.put('/labor-force-participation/:date', async (req, res) => {
    const { date } = req.params;
    const { value } = req.body;
    try {
        const result = await updateLaborForceParticipationDataByDate(date, value);
        if (!result) {
            return res.status(404).json({ error: 'Labor Force Participation data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to update Labor Force Participation data' });
    }
});

// Delete Labor Force Participation entry by date
laborForceParticipationRouter.delete('/labor-force-participation/:date', async (req, res) => {
    const { date } = req.params;
    try {
        const result = await deleteLaborForceParticipationDataByDate(date);
        if (!result) {
            return res.status(404).json({ error: 'Labor Force Participation data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to delete Labor Force Participation data' });
    }
});

module.exports = laborForceParticipationRouter;