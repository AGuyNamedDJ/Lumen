const express = require('express');
const { fetchVIXData } = require('../../dataFetching/yahooFinance/fetchVIXData');
const router = express.Router();

// Route to manually trigger the VIX data fetch
router.get('/fetch-vix', async (req, res) => {
    try {
        await fetchVIXData();
        res.status(200).json({ message: 'VIX data fetched and stored successfully' });
    } catch (error) {
        console.error('Error in fetching VIX data:', error);
        res.status(500).json({ error: 'Error fetching VIX data' });
    }
});

module.exports = router;