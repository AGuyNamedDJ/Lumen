const cron = require('node-cron');
const { fetchAggregatesData } = require('../fetch/aggregates');

function scheduleAggregatesDataUpdates() {
    cron.schedule('0 7 * * 1-5', async () => {  // This will run at 7 AM every weekday (Monday to Friday)
        console.log("Fetching daily Aggregates Data for SPY...");
        await fetchAggregatesData('SPY', 1, 'day', 'YYYY-MM-DD', 'YYYY-MM-DD');
        console.log("Aggregates Data for SPY fetched and processed.");
    });
}

module.exports = { scheduleAggregatesDataUpdates };