const cron = require('node-cron');
const { fetchGDPData } = require('../historic/gdpData');

function scheduleGDPUpdates() {
    // Schedule the job for the last day of the month at 6:00 PM ET, every quarter (March, June, September, December)
    cron.schedule('0 18 31 3,6,9,12 *', async () => {
        await fetchGDPData();
        console.log('GDP data updated.');
    });
}

module.exports = { scheduleGDPUpdates };