const cron = require('node-cron');
const { fetchAverageHourlyEarningsData } = require('../historic/averageHourlyEarningsData');

function scheduleAverageHourlyEarningsUpdates() {
    cron.schedule('0 18 1-7 * 5', async () => {
        // This schedules the job for the first Friday of every month at 6:00 PM
        await fetchAverageHourlyEarningsData();
        console.log('Average Hourly Earnings data updated.');
    });}

module.exports = { scheduleAverageHourlyEarningsUpdates };