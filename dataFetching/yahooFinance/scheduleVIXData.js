const cron = require('node-cron');
const { fetchVIXData } = require('./fetchVIXData');

function scheduleVIXUpdates() {
    cron.schedule('*/30 7-16 * * 1-5', async () => {
        // This schedules the job to run every 30 minutes from 7 AM to 4 PM, Monday to Friday
        await fetchVIXData();
        console.log('VIX data updated.');
    });
}

module.exports = { scheduleVIXUpdates };