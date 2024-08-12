const cron = require('node-cron');
const { fetchAndStoreSMAData } = require('../fetch/smaIndicators');

// Define the symbols and their respective SMA windows
const symbols = ['SPY']; 
const windows = [20, 50, 100, 200]; 

function scheduleSMAUpdates() {
    // Fetch daily SMA data every day at 7 AM
    cron.schedule('0 7 * * 1-5', async () => {  // Runs at 7 AM, Monday through Friday
        try {
            console.log("Starting daily SMA data fetch...");
            for (const symbol of symbols) {
                for (const window of windows) {
                    await fetchAndStoreSMAData(symbol, 'day', window, 'close');
                    console.log(`Daily SMA data for ${symbol} (window: ${window}) fetched and processed.`);
                }
            }
        } catch (error) {
            console.error('Failed to fetch daily SMA data:', error);
        }
    }, {
        timezone: "America/Chicago"
    });

    console.log('Scheduled daily SMA updates for SPY at 7 AM CT, Monday through Friday.');
}

module.exports = { scheduleSMAUpdates };