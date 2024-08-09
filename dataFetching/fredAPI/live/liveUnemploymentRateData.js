const cron = require('node-cron');
const { fetchUnemploymentRateData } = require('../historic/unemploymentRateData');

function scheduleUnemploymentRateUpdates() {
    cron.schedule('0 23 8 * *', async () => {
        // This schedules the job for the 8th of every month at 11:00 PM
        await fetchUnemploymentRateData();
        console.log('Unemployment Rate data updated.');
    });
}

module.exports = { scheduleUnemploymentRateUpdates };