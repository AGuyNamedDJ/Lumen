const cron = require('node-cron');
const { fetchPCEData } = require('../historic/pceData');

function schedulePCEUpdates() {
    cron.schedule('0 23 30 * *', async () => {
        // This schedules the job for the 30th of every month at 11:00 PM
        await fetchPCEData();
        console.log('PCE data updated.');
    });
}

module.exports = { schedulePCEUpdates };