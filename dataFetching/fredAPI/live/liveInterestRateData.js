const cron = require('node-cron');
const { fetchInterestRateData } = require('../historic/interestRateData');

function scheduleInterestRateUpdates() {
    // Schedule the job for the 28th of every month at 12:00 PM
    cron.schedule('0 12 28 * *', async () => {
        // You can specify different series IDs if needed
        const series_ids = ['FEDFUNDS', 'DFF']; // Example series IDs for interest rates

        for (const series_id of series_ids) {
            await fetchInterestRateData(series_id);
        }

        console.log('Interest Rate data updated.');
    });
}

module.exports = { scheduleInterestRateUpdates };