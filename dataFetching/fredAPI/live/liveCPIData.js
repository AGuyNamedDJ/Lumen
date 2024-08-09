const cron = require('node-cron');
const { fetchCPIData } = require('../../../api/fredAPI/fredCPIData');

function scheduleCPIUpdates() {
    cron.schedule('59 23 15 * *', async () => {
        // This schedules the job for the 15th of every month at 11:59 PM
        await fetchCPIData();
        console.log('CPI data updated.');
    });
}

module.exports = { scheduleCPIUpdates };