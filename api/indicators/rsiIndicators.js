const express = require('express');
const { storeRSIData, getAllRSIIndicators, getRSIIndicatorById, getRSIIndicatorsBySymbol, deleteRSIIndicator } = require('../../db/indicators/rsiIndicators');
const { getTechnicalIndicator } = require('../polygonApi/polygonApi');
require('dotenv').config();

const rsiRouter = express.Router();

// Route to fetch live RSI indicator from Polygon
rsiRouter.get('/live', async (req, res) => {
    const { symbol, timespan, window, series_type } = req.query;

    if (!symbol || !timespan || !window || !series_type) {
        return res.status(400).json({ error: 'All query parameters (symbol, timespan, window, series_type) are required' });
    }

    try {
        const indicatorData = await getTechnicalIndicator(symbol, 'rsi', timespan, window, series_type);
        
        if (indicatorData && indicatorData.results && Array.isArray(indicatorData.results.values)) {
            const formattedData = indicatorData.results.values.map(dataPoint => ({
                symbol,
                period: window,
                timespan,
                timestamp: dataPoint.timestamp || dataPoint[0], // Adjust according to data structure, same for rest
                value: dataPoint.value || dataPoint[1] 
            }));

            await storeRSIData(formattedData);
            res.json(indicatorData);
        } else {
            res.status(500).json({ error: 'Unexpected response format from Polygon API' });
        }
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch and store live RSI indicator' });
    }
});

// Route to fetch all RSI indicators
rsiRouter.get('/', async (req, res) => {
    try {
        const indicators = await getAllRSIIndicators();
        res.json(indicators);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch RSI indicators' });
    }
});

// Route to fetch RSI indicator by ID
rsiRouter.get('/:id', async (req, res) => {
    const { id } = req.params;
    try {
        const indicator = await getRSIIndicatorById(id);
        res.json(indicator);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch RSI indicator' });
    }
});

// Route to fetch RSI indicators by symbol
rsiRouter.get('/symbol/:symbol', async (req, res) => {
    const { symbol } = req.params;
    try {
        const indicators = await getRSIIndicatorsBySymbol(symbol);
        res.json(indicators);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch RSI indicators by symbol' });
    }
});

// Route to create a new RSI indicator
rsiRouter.post('/', async (req, res) => {
    const { symbol, period, timespan, timestamp, value } = req.body;
    try {
        const newIndicator = await storeRSIData([{ symbol, period, timespan, timestamp, value }]);
        res.json(newIndicator);
    } catch (error) {
        res.status(500).json({ error: 'Failed to create RSI indicator' });
    }
});

// Route to delete an RSI indicator
rsiRouter.delete('/:id', async (req, res) => {
    const { id } = req.params;
    try {
        const deletedIndicator = await deleteRSIIndicator(id);
        res.json(deletedIndicator);
    } catch (error) {
        res.status(500).json({ error: 'Failed to delete RSI indicator' });
    }
});

module.exports = rsiRouter;