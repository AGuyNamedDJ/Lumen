const cron = require('node-cron');
const { fetchLaborForceParticipationRateData } = require('../historic/laborForceParticipationRateData');

function scheduleLaborForceParticipationRateUpdates() {
    cron.schedule('0 12 1-7 * 5', async () => {
        // This schedules the job for the first Friday of every month at 12:00 PM (noon)
        await fetchLaborForceParticipationRateData();
        console.log('Labor Force Participation Rate data updated.');
    });
}

module.exports = { scheduleLaborForceParticipationRateUpdates };