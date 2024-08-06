const express = require('express');
const { storeMACDData, getAllMACDIndicators, getMACDIndicatorById, getMACDIndicatorsBySymbol, deleteMACDIndicator } = require('../../db/indicators/macdIndicators');
const { getTechnicalIndicator } = require('../polygonApi/polygonApi');
require('dotenv').config();

const macdRouter = express.Router();

// Route to fetch live MACD indicator from Polygon
macdRouter.get('/live', async (req, res) => {
    const { symbol, timespan, window, series_type } = req.query;

    if (!symbol || !timespan || !window || !series_type) {
        return res.status(400).json({ error: 'All query parameters (symbol, timespan, window, series_type) are required' });
    }

    try {
        const indicatorData = await getTechnicalIndicator(symbol, 'macd', timespan, window, series_type);
        
        if (indicatorData && indicatorData.results && Array.isArray(indicatorData.results.values)) {
            const formattedData = indicatorData.results.values.map(dataPoint => ({
                symbol,
                period: window,
                timespan,
                timestamp: dataPoint.timestamp || dataPoint[0],
                macd_line: dataPoint.macd_line || dataPoint[1] || 0, 
                signal_line: dataPoint.signal_line || dataPoint[2] || 0, 
                histogram: dataPoint.histogram || dataPoint[3] || 0
            }));

            await storeMACDData(formattedData);
            res.json(indicatorData);
        } else {
            res.status(500).json({ error: 'Unexpected response format from Polygon API' });
        }
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch and store live MACD indicator' });
    }
});

// Route to fetch all MACD indicators
macdRouter.get('/', async (req, res) => {
    try {
        const indicators = await getAllMACDIndicators();
        res.json(indicators);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch MACD indicators' });
    }
});

// Route to fetch MACD indicator by ID
macdRouter.get('/:id', async (req, res) => {
    const { id } = req.params;
    try {
        const indicator = await getMACDIndicatorById(id);
        res.json(indicator);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch MACD indicator' });
    }
});

// Route to fetch MACD indicators by symbol
macdRouter.get('/symbol/:symbol', async (req, res) => {
    const { symbol } = req.params;
    try {
        const indicators = await getMACDIndicatorsBySymbol(symbol);
        res.json(indicators);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch MACD indicators by symbol' });
    }
});

// Route to create a new MACD indicator
macdRouter.post('/', async (req, res) => {
    const { symbol, period, timespan, timestamp, macd_line, signal_line, histogram } = req.body;
    try {
        const newIndicator = await storeMACDData([{ symbol, period, timespan, timestamp, macd_line, signal_line, histogram }]);
        res.json(newIndicator);
    } catch (error) {
        res.status(500).json({ error: 'Failed to create MACD indicator' });
    }
});

// Route to delete a MACD indicator
macdRouter.delete('/:id', async (req, res) => {
    const { id } = req.params;
    try {
        const deletedIndicator = await deleteMACDIndicator(id);
        res.json(deletedIndicator);
    } catch (error) {
        res.status(500).json({ error: 'Failed to delete MACD indicator' });
    }
});

module.exports = macdRouter;