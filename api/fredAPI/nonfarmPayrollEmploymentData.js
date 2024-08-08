const express = require('express');
require("dotenv").config();
const nonfarmPayrollEmploymentRouter = express.Router();
const {
    storeNonfarmPayrollEmploymentData,
    getAllNonfarmPayrollEmploymentData,
    getNonfarmPayrollEmploymentDataByDate,
    updateNonfarmPayrollEmploymentDataByDate,
    deleteNonfarmPayrollEmploymentDataByDate,
} = require('../../db/fredAPI/nonfarmPayrollEmploymentData');
const axios = require('axios');

// Series ID for Nonfarm Payroll Employment
const seriesID = 'PAYEMS';

// Fetch and store Nonfarm Payroll Employment data from FRED API
const fetchNonfarmPayrollEmploymentData = async () => {
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

            await storeNonfarmPayrollEmploymentData({ date, value });
        }
    } catch (error) {
        console.error('Error fetching Nonfarm Payroll Employment data:', error);
    }
};

// Route to fetch Nonfarm Payroll Employment data from FRED and store in DB
nonfarmPayrollEmploymentRouter.get('/fetch-nonfarm-payroll-employment', async (req, res) => {
    try {
        await fetchNonfarmPayrollEmploymentData();
        res.status(200).json({ message: 'Nonfarm Payroll Employment data fetched and stored successfully.' });
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch and store Nonfarm Payroll Employment data.' });
    }
});

// Create new Nonfarm Payroll Employment entry
nonfarmPayrollEmploymentRouter.post('/nonfarm-payroll-employment', async (req, res) => {
    const { date, value } = req.body;
    try {
        const newEntry = await storeNonfarmPayrollEmploymentData({ date, value });
        res.status(201).json(newEntry);
    } catch (error) {
        res.status(500).json({ error: 'Failed to create Nonfarm Payroll Employment data entry' });
    }
});

// Read all Nonfarm Payroll Employment entries
nonfarmPayrollEmploymentRouter.get('/nonfarm-payroll-employment', async (req, res) => {
    try {
        const result = await getAllNonfarmPayrollEmploymentData();
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch Nonfarm Payroll Employment data' });
    }
});

// Read single Nonfarm Payroll Employment entry by date
nonfarmPayrollEmploymentRouter.get('/nonfarm-payroll-employment/:date', async (req, res) => {
    const { date } = req.params;
    try {
        const result = await getNonfarmPayrollEmploymentDataByDate(date);
        if (!result) {
            return res.status(404).json({ error: 'Nonfarm Payroll Employment data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch Nonfarm Payroll Employment data' });
    }
});

// Update Nonfarm Payroll Employment entry by date
nonfarmPayrollEmploymentRouter.put('/nonfarm-payroll-employment/:date', async (req, res) => {
    const { date } = req.params;
    const { value } = req.body;
    try {
        const result = await updateNonfarmPayrollEmploymentDataByDate(date, value);
        if (!result) {
            return res.status(404).json({ error: 'Nonfarm Payroll Employment data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to update Nonfarm Payroll Employment data' });
    }
});

// Delete Nonfarm Payroll Employment entry by date
nonfarmPayrollEmploymentRouter.delete('/nonfarm-payroll-employment/:date', async (req, res) => {
    const { date } = req.params;
    try {
        const result = await deleteNonfarmPayrollEmploymentDataByDate(date);
        if (!result) {
            return res.status(404).json({ error: 'Nonfarm Payroll Employment data not found' });
        }
        res.json(result);
    } catch (error) {
        res.status(500).json({ error: 'Failed to delete Nonfarm Payroll Employment data' });
    }
});

module.exports = nonfarmPayrollEmploymentRouter;