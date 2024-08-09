const cron = require('node-cron');
const { fetchIndustrialProductionData } = require('../historic/industrialProductionData');

function scheduleIndustrialProductionUpdates() {
    // Schedule the job for the 17th of every month at 12:00 PM
    cron.schedule('0 12 17 * *', async () => {
        await fetchIndustrialProductionData();
        console.log('Industrial Production data updated.');
    });
}

module.exports = { scheduleIndustrialProductionUpdates };