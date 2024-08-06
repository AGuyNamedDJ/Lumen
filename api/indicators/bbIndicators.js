const express = require('express');
const { storeBBData, getAllBBIndicators, getBBIndicatorById, getBBIndicatorsBySymbol, deleteBBIndicator } = require('../../db/indicators/bbIndicators');
const { getTechnicalIndicator } = require('../polygonApi/polygonApi');
require('dotenv').config();

const bbRouter = express.Router();

// Route to fetch live BB indicator from Polygon
bbRouter.get('/live', async (req, res) => {
    const { symbol, timespan, window, series_type } = req.query;

    if (!symbol || !timespan || !window || !series_type) {
        return res.status(400).json({ error: 'All query parameters (symbol, timespan, window, series_type) are required' });
    }

    try {
        const indicatorData = await getTechnicalIndicator(symbol, 'bb', timespan, window, series_type);
        
        if (indicatorData && indicatorData.results && Array.isArray(indicatorData.results.values)) {
            const formattedData = indicatorData.results.values.map(dataPoint => ({
                symbol,
                period: window,
                timespan,
                timestamp: dataPoint.timestamp || dataPoint[0],
                middle_band: dataPoint.value || dataPoint[1],
                upper_band: dataPoint.upper_band,
                lower_band: dataPoint.lower_band
            }));

            await storeBBData(formattedData);
            res.json(indicatorData);
        } else {
            res.status(500).json({ error: 'Unexpected response format from Polygon API' });
        }
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch and store live BB indicator' });
    }
});

// Route to fetch all BB indicators
bbRouter.get('/', async (req, res) => {
    try {
        const indicators = await getAllBBIndicators();
        res.json(indicators);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch BB indicators' });
    }
});

// Route to fetch BB indicator by ID
bbRouter.get('/:id', async (req, res) => {
    const { id } = req.params;
    try {
        const indicator = await getBBIndicatorById(id);
        res.json(indicator);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch BB indicator' });
    }
});

// Route to fetch BB indicators by symbol
bbRouter.get('/symbol/:symbol', async (req, res) => {
    const { symbol } = req.params;
    try {
        const indicators = await getBBIndicatorsBySymbol(symbol);
        res.json(indicators);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch BB indicators by symbol' });
    }
});

// Route to create a new BB indicator
bbRouter.post('/', async (req, res) => {
    const { symbol, period, timespan, timestamp, middle_band, upper_band, lower_band } = req.body;
    try {
        const newIndicator = await storeBBData([{ symbol, period, timespan, timestamp, middle_band, upper_band, lower_band }]);
        res.json(newIndicator);
    } catch (error) {
        res.status(500).json({ error: 'Failed to create BB indicator' });
    }
});

// Route to delete a BB indicator
bbRouter.delete('/:id', async (req, res) => {
    const { id } = req.params;
    try {
        const deletedIndicator = await deleteBBIndicator(id);
        res.json(deletedIndicator);
    } catch (error) {
        res.status(500).json({ error: 'Failed to delete BB indicator' });
    }
});

module.exports = bbRouter;