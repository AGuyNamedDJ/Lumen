const express = require('express');
const { storeEMAData, getAllEMAIndicators, getEMAIndicatorById, getEMAIndicatorsBySymbol, deleteEMAIndicator } = require('../../db/indicators/emaIndicators');
const { getTechnicalIndicator } = require('../polygonApi/polygonApi');
require('dotenv').config();

const emaRouter = express.Router();

// Route to fetch live EMA indicator from Polygon
emaRouter.get('/live', async (req, res) => {
    const { symbol, timespan, window, series_type } = req.query;

    if (!symbol || !timespan || !window || !series_type) {
        return res.status(400).json({ error: 'All query parameters (symbol, timespan, window, series_type) are required' });
    }

    try {
        const indicatorData = await getTechnicalIndicator(symbol, 'ema', timespan, window, series_type);
        
        if (indicatorData && indicatorData.results && Array.isArray(indicatorData.results.values)) {
            const formattedData = indicatorData.results.values.map(dataPoint => ({
                symbol,
                period: window,
                timespan,
                timestamp: dataPoint.timestamp || dataPoint[0], 
                value: dataPoint.value || dataPoint[1] 
            }));

            await storeEMAData(formattedData);
            res.json(indicatorData);
        } else {
            res.status(500).json({ error: 'Unexpected response format from Polygon API' });
        }
    } catch (error) {
        console.error('Error fetching and storing live EMA indicator:', error);
        res.status(500).json({ error: 'Failed to fetch and store live EMA indicator' });
    }
});

// Route to fetch all EMA indicators
emaRouter.get('/', async (req, res) => {
    try {
        const indicators = await getAllEMAIndicators();
        res.json(indicators);
    } catch (error) {
        console.error('Error fetching all EMA indicators:', error);
        res.status(500).json({ error: 'Failed to fetch EMA indicators' });
    }
});

// Route to fetch EMA indicator by ID
emaRouter.get('/:id', async (req, res) => {
    const { id } = req.params;
    try {
        const indicator = await getEMAIndicatorById(id);
        res.json(indicator);
    } catch (error) {
        console.error('Error fetching EMA indicator by ID:', error);
        res.status(500).json({ error: 'Failed to fetch EMA indicator' });
    }
});

// Route to fetch EMA indicators by symbol
emaRouter.get('/symbol/:symbol', async (req, res) => {
    const { symbol } = req.params;
    try {
        const indicators = await getEMAIndicatorsBySymbol(symbol);
        res.json(indicators);
    } catch (error) {
        console.error('Error fetching EMA indicators by symbol:', error);
        res.status(500).json({ error: 'Failed to fetch EMA indicators by symbol' });
    }
});

// Route to create a new EMA indicator
emaRouter.post('/', async (req, res) => {
    const { symbol, period, timespan, timestamp, value } = req.body;
    try {
        const newIndicator = await storeEMAData([{ symbol, period, timespan, timestamp, value }]);
        res.json(newIndicator);
    } catch (error) {
        console.error('Error creating EMA indicator:', error);
        res.status(500).json({ error: 'Failed to create EMA indicator' });
    }
});

// Route to delete an EMA indicator
emaRouter.delete('/:id', async (req, res) => {
    const { id } = req.params;
    try {
        const deletedIndicator = await deleteEMAIndicator(id);
        res.json(deletedIndicator);
    } catch (error) {
        console.error('Error deleting EMA indicator:', error);
        res.status(500).json({ error: 'Failed to delete EMA indicator' });
    }
});

module.exports = emaRouter;