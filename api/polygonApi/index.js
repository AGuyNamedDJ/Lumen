const express = require('express');
const { getStockQuote, getTechnicalIndicator, getHistoricalAggregates, searchSymbols } = require('./polygonApi');
require('dotenv').config();

const polygonRouter = express.Router();

// Route to fetch stock quote
polygonRouter.get('/stock-quote', async (req, res) => {
    const { symbol } = req.query;
    if (!symbol) {
        return res.status(400).json({ error: 'Symbol query parameter is required' });
    }
    try {
        const quote = await getStockQuote(symbol);
        res.json(quote);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch stock quote' });
    }
});

// Route to fetch technical indicator
polygonRouter.get('/technical-indicator', async (req, res) => {
    const { symbol, indicator, timespan, window, series_type } = req.query;

    console.log('Received technical indicator request with query parameters:', { symbol, indicator, timespan, window, series_type });

    if (!symbol || !indicator || !timespan || !window || !series_type) {
        return res.status(400).json({ error: 'All query parameters (symbol, indicator, timespan, window, series_type) are required' });
    }

    try {
        const indicatorData = await getTechnicalIndicator(symbol, indicator, timespan, window, series_type);
        res.json(indicatorData);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch technical indicator' });
    }
});

// Route to fetch historical aggregates
polygonRouter.get('/historical-aggregates', async (req, res) => {
    const { symbol, from, to, timespan } = req.query;

    console.log('Received historical aggregates request with query parameters:', { symbol, from, to, timespan });

    if (!symbol || !from || !to || !timespan) {
        return res.status(400).json({ error: 'All query parameters (symbol, from, to, timespan) are required' });
    }

    try {
        const aggregates = await getHistoricalAggregates(symbol, from, to, timespan);
        res.json(aggregates);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch historical aggregates' });
    }
});

// Route to search for symbols
polygonRouter.get('/search-symbols', async (req, res) => {
    const { query } = req.query;

    if (!query) {
        return res.status(400).json({ error: 'Query parameter is required' });
    }

    try {
        const symbols = await searchSymbols(query);
        res.json(symbols);
    } catch (error) {
        res.status(500).json({ error: 'Failed to search for symbols' });
    }
});

module.exports = polygonRouter;