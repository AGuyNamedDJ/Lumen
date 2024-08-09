const express = require('express');
require("dotenv").config();
const averageHourlyEarningsRouter = express.Router();
const {
    storeAverageHourlyEarningsData,
    getAllAverageHourlyEarningsData,
    getAverageHourlyEarningsDataByDate,
    updateAverageHourlyEarningsDataByDate,
    deleteAverageHourlyEarningsDataByDate,
} = require('../../db/fredAPI/averageHourlyEarningsData');
const axios = require('axios');

// Series ID for Average Hourly Earnings
const seriesID = 'CES0500000003';

// Route to fetch Average Hourly Earnings data from FRED and store in DB
averageHourlyEarningsRouter.get('/fetch-average-hourly-earnings', async (req, res) => {
    try {
        await fetchAverageHourlyEarningsData();
        res.status(200).json({ message: 'Average Hourly Earnings data fetched and stored successfully.' });
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch and store Average Hourly Earnings data.' });
    }
});

// Create new Average Hourly Earnings entry
averageHourlyEarningsRouter.post('/average-hourly-earnings', async (req, res) => {
    const { date, value } = req.body;
    try {
        const newEntry = await storeAverageHourlyEarningsData({ date, value });
        res.status(201).json(newEntry);
    } catch (error) {
        res.status(500).json({ error: 'Failed to create Average Hourly Earnings data entry' });
    }
});

// Read all Average Hourly Earnings entries
averageHourlyEarningsRouter.get('/average-hourly-earnings', async (req, res) => {
    try {
        const result = await getAllAverageHourlyEarningsData();
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch Average Hourly Earnings data' });
    }
});

// Read single Average Hourly Earnings entry by date
averageHourlyEarningsRouter.get('/average-hourly-earnings/:date', async (req, res) => {
    const { date } = req.params;
    try {
        const result = await getAverageHourlyEarningsDataByDate(date);
        if (!result) {
            return res.status(404).json({ error: 'Average Hourly Earnings data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch Average Hourly Earnings data' });
    }
});

// Update Average Hourly Earnings entry by date
averageHourlyEarningsRouter.put('/average-hourly-earnings/:date', async (req, res) => {
    const { date } = req.params;
    const { value } = req.body;
    try {
        const result = await updateAverageHourlyEarningsDataByDate(date, value);
        if (!result) {
            return res.status(404).json({ error: 'Average Hourly Earnings data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to update Average Hourly Earnings data' });
    }
});

// Delete Average Hourly Earnings entry by date
averageHourlyEarningsRouter.delete('/average-hourly-earnings/:date', async (req, res) => {
    const { date } = req.params;
    try {
        const result = await deleteAverageHourlyEarningsDataByDate(date);
        if (!result) {
            return res.status(404).json({ error: 'Average Hourly Earnings data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to delete Average Hourly Earnings data' });
    }
});

module.exports = averageHourlyEarningsRouter;
