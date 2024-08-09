const cron = require('node-cron');
const { fetchNonfarmPayrollEmploymentData } = require('../historic/nonfarmPayrollEmploymentData');

function scheduleNonfarmPayrollEmploymentUpdates() {
    cron.schedule('0 18 1-7 * 5', async () => {
        // This schedules the job for the first Friday of every month at 6:00 PM
        await fetchNonfarmPayrollEmploymentData();
        console.log('Nonfarm Payroll Employment data updated.');
    });
}

module.exports = { scheduleNonfarmPayrollEmploymentUpdates };