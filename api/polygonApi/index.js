const express = require('express');
const { getStockQuote, getHistoricalAggregates } = require('./polygonApi');
require('dotenv').config();

const polygonRouter = express.Router();

polygonRouter.get('/stock-quote', async (req, res) => {
    const { symbol } = req.query;
    try {
        const quote = await getStockQuote(symbol);
        res.json(quote);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch stock quote' });
    }
});

polygonRouter.get('/historical-aggregates', async (req, res) => {
    const { symbol, from, to } = req.query;
    try {
        const aggregates = await getHistoricalAggregates(symbol, from, to);
        res.json(aggregates);
    } catch (error) {
        res.status(500).json({ error: 'Failed to fetch historical aggregates' });
    }
});

module.exports = polygonRouter;