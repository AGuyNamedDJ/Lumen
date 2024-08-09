const cron = require('node-cron');
const { fetchCoreInflationData } = require('../historic/coreInflationData');

function scheduleCoreInflationUpdates() {
    cron.schedule('0 11 15 * *', async () => {
        // This schedules the job for the 15th of every month at 11:00 AM
        await fetchCoreInflationData();
        console.log('Core Inflation data updated.');
    });
}

module.exports = { scheduleCoreInflationUpdates };