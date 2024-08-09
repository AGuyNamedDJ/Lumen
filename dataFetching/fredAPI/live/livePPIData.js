const cron = require('node-cron');
const { fetchPPIData } = require('../historic/ppiData');

function schedulePPIUpdates() {
    cron.schedule('0 23 15 * *', async () => {
        // This schedules the job for the 15th of every month at 11:00 PM
        await fetchPPIData();
        console.log('PPI data updated.');
    });
}

module.exports = { schedulePPIUpdates };