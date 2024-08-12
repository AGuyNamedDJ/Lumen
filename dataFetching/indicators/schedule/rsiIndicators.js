const cron = require('node-cron');
const { fetchAndStoreRSIData } = require('../fetch/rsiIndicators');

// Define the symbols and their respective window periods
const symbols = ['SPY']; 
const windows = [14]; 

function scheduleRSIUpdates() {
    // Fetch daily RSI data every day at 7 AM
    cron.schedule('0 7 * * 1-5', async () => {  // Runs at 7 AM, Monday through Friday
        try {
            for (const symbol of symbols) {
                await fetchAndStoreRSIData(symbol, windows[0], 'close');
                console.log(`Daily RSI data for ${symbol} fetched and processed.`);
            }
        } catch (error) {
            console.error('Failed to fetch daily RSI data:', error);
        }
    }, {
        timezone: "America/Chicago"
    });

    console.log('Scheduled daily RSI updates for SPY at 7 AM CT, Monday through Friday.');
}

module.exports = { scheduleRSIUpdates };