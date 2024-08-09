const cron = require('node-cron');
const { fetchConsumerConfidenceData } = require('../historic/consumerConfidenceData');

function scheduleConsumerConfidenceUpdates() {
    cron.schedule('0 11 * * 2', async () => {
        const today = new Date();
        const lastTuesdayOfMonth = new Date(today.getFullYear(), today.getMonth() + 1, 0);
        
        // Adjust to last Tuesday of the month
        lastTuesdayOfMonth.setDate(lastTuesdayOfMonth.getDate() - ((lastTuesdayOfMonth.getDay() + 4) % 7));

        if (today.getDate() === lastTuesdayOfMonth.getDate()) {
            await fetchConsumerConfidenceData();
            console.log('Consumer Confidence data updated.');
        }
    });
}

module.exports = { scheduleConsumerConfidenceUpdates };