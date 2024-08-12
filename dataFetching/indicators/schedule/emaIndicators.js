const cron = require('node-cron');
const { fetchEMAData } = require('../fetch/emaIndicators');

function scheduleEMAUpdates() {
    const symbols = ['SPY']; 
    const windows = [1, 20]; 

    cron.schedule('0 7 * * 1-5', async () => {  // This will run at 7 AM every weekday (Monday to Friday)
        console.log('Starting daily EMA data fetch...');
        for (const symbol of symbols) {
            for (const window of windows) {
                try {
                    await fetchEMAData(symbol, window, 'close');
                } catch (error) {
                    console.error(`Failed to fetch daily EMA data for ${symbol} (window: ${window}):`, error);
                }
            }
        }
        console.log('Daily EMA data fetch completed.');
    }, {
        timezone: "America/Chicago"
    });

    console.log('Daily EMA Updates scheduled at 7 AM CT, Monday through Friday.');
}

module.exports = { scheduleEMAUpdates };