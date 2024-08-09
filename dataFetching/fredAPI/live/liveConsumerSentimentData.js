const cron = require('node-cron');
const { fetchConsumerSentimentData } = require('../../../api/fredAPI/consumerSentimentData');

function scheduleConsumerSentimentUpdates() {
    cron.schedule('0 11 * * 2L', async () => {
        // This schedules the job for the last Tuesday of every month at 11:00 AM ET
        await fetchConsumerSentimentData();
        console.log('Consumer Sentiment data updated.');
    });
}

module.exports = { scheduleConsumerSentimentUpdates };