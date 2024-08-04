const express = require('express');
const { storeSMAData, getAllSMAIndicators, getSMAIndicatorById, getSMAIndicatorsBySymbol, deleteSMAIndicator } = require('../../db/indicators/smaIndicators');
const { getTechnicalIndicator } = require('../polygonApi/polygonApi');
require('dotenv').config();

const smaRouter = express.Router();

// Route to fetch live SMA indicator from Polygon
smaRouter.get('/live', async (req, res) => {
    const { symbol, timespan, window, series_type } = req.query;

    if (!symbol || !timespan || !window || !series_type) {
        return res.status(400).json({ error: 'All query parameters (symbol, timespan, window, series_type) are required' });
    }

    try {
        const indicatorData = await getTechnicalIndicator(symbol, 'sma', timespan, window, series_type);
        
        if (indicatorData && indicatorData.results && Array.isArray(indicatorData.results.values)) {
            const formattedData = indicatorData.results.values.map(dataPoint => ({
                symbol,
                period: window,
                timespan,
                timestamp: dataPoint.timestamp || dataPoint[0], 
                value: dataPoint.value || dataPoint[1] 
            }));

            await storeSMAData(formattedData);
            res.json(indicatorData);
        } else {
            res.status(500).json({ error: 'Unexpected response format from Polygon API' });
        }
    } catch (error) {
        console.error('Error fetching and storing live SMA indicator:', error);
        res.status(500).json({ error: 'Failed to fetch and store live SMA indicator' });
    }
});

// Route to fetch all SMA indicators
smaRouter.get('/', async (req, res) => {
    try {
        const indicators = await getAllSMAIndicators();
        res.json(indicators);
    } catch (error) {
        console.error('Error fetching all SMA indicators:', error);
        res.status(500).json({ error: 'Failed to fetch SMA indicators' });
    }
});

// Route to fetch SMA indicator by ID
smaRouter.get('/:id', async (req, res) => {
    const { id } = req.params;
    try {
        const indicator = await getSMAIndicatorById(id);
        res.json(indicator);
    } catch (error) {
        console.error('Error fetching SMA indicator by ID:', error);
        res.status(500).json({ error: 'Failed to fetch SMA indicator' });
    }
});

// Route to fetch SMA indicators by symbol
smaRouter.get('/symbol/:symbol', async (req, res) => {
    const { symbol } = req.params;
    try {
        const indicators = await getSMAIndicatorsBySymbol(symbol);
        res.json(indicators);
    } catch (error) {
        console.error('Error fetching SMA indicators by symbol:', error);
        res.status(500).json({ error: 'Failed to fetch SMA indicators by symbol' });
    }
});

// Route to create a new SMA indicator
smaRouter.post('/', async (req, res) => {
    const { symbol, period, timespan, timestamp, value } = req.body;
    try {
        const newIndicator = await storeSMAData([{ symbol, period, timespan, timestamp, value }]);
        res.json(newIndicator);
    } catch (error) {
        console.error('Error creating SMA indicator:', error);
        res.status(500).json({ error: 'Failed to create SMA indicator' });
    }
});

// Route to delete an SMA indicator
smaRouter.delete('/:id', async (req, res) => {
    const { id } = req.params;
    try {
        const deletedIndicator = await deleteSMAIndicator(id);
        res.json(deletedIndicator);
    } catch (error) {
        console.error('Error deleting SMA indicator:', error);
        res.status(500).json({ error: 'Failed to delete SMA indicator' });
    }
});

module.exports = smaRouter;