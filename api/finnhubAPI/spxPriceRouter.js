const express = require('express');
const { getRealTimeQuotes } = require('./finnhubAPI');

const router = express.Router();

// 
router.get('/spx-price', (req, res) => {
    getRealTimeQuotes('SPY', (error, data) => {
        if (error) {
            return res.status(500).json({ error: 'Failed to fetch SPX price' });
        }
        res.json({ current_price: data.c });
    });
});

module.exports = router;