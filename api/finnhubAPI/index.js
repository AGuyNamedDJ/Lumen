const express = require('express');
const finnhubRouter = express.Router();
const { getRealTimeQuotes } = require('./finnhubAPI');

// Ex
finnhubRouter.get('/quote/:symbol', (req, res) => {
    const symbol = req.params.symbol;
    getRealTimeQuotes(symbol, (error, data) => {
        if (error) {
            return res.status(500).json({ success: false, message: error.message });
        }
        res.json({ success: true, data });
    });
});

module.exports = finnhubRouter;