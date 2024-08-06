const express = require('express');
const { storeAggregatesData, getAllAggregates, getAggregatesBySymbol, getAggregatesBySymbolAndTimespan, deleteAggregates } = require('../../db/indicators/aggregates');
const { getHistoricalAggregates } = require('../polygonApi/polygonApi');
require('dotenv').config();

const aggregatesRouter = express.Router();

// Route to fetch live aggregates from Polygon
aggregatesRouter.get('/live', async (req, res) => {
    const { symbol, multiplier, timespan, from, to } = req.query;

    if (!symbol || !multiplier || !timespan || !from || !to) {
        return res.status(400).json({ error: 'All query parameters (symbol, multiplier, timespan, from, to) are required' });
    }

    try {
        const aggregatesData = await getHistoricalAggregates(symbol, from, to, timespan);
        
        if (aggregatesData && aggregatesData.results && Array.isArray(aggregatesData.results)) {
            const formattedData = aggregatesData.results.map(dataPoint => ({
                symbol,
                multiplier: parseInt(multiplier),
                timespan,
                timestamp: dataPoint.t,
                open: dataPoint.o,
                high: dataPoint.h,
                low: dataPoint.l,
                close: dataPoint.c,
                volume: dataPoint.v
            }));

            await storeAggregatesData(formattedData);
            res.json(aggregatesData);
        } else {
            res.status(500).json({ error: 'Unexpected response format from Polygon API' });
        }
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch and store live aggregates' });
    }
});

// Route to fetch all aggregates
aggregatesRouter.get('/', async (req, res) => {
    try {
        const aggregates = await getAllAggregates();
        res.json(aggregates);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch aggregates' });
    }
});

// Route to fetch aggregates by symbol
aggregatesRouter.get('/symbol/:symbol', async (req, res) => {
    const { symbol } = req.params;
    try {
        const aggregates = await getAggregatesBySymbol(symbol);
        res.json(aggregates);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch aggregates by symbol' });
    }
});

// Route to fetch aggregates by symbol and timespan
aggregatesRouter.get('/symbol/:symbol/:timespan', async (req, res) => {
    const { symbol, timespan } = req.params;
    try {
        const aggregates = await getAggregatesBySymbolAndTimespan(symbol, timespan);
        res.json(aggregates);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch aggregates by symbol and timespan' });
    }
});

// Route to delete aggregates
aggregatesRouter.delete('/:id', async (req, res) => {
    const { id } = req.params;
    try {
        const deletedAggregate = await deleteAggregates(id);
        res.json(deletedAggregate);
    } catch (error) {
        res.status(500).json({ error: 'Failed to delete aggregates' });
    }
});

module.exports = aggregatesRouter;