const cron = require('node-cron');
const { fetchMACDData } = require('../fetch/macdIndicators');

function scheduleMACDUpdates() {
    const symbol = 'SPY'; 
    
    // Fetch daily MACD data every day at 7 AM
    cron.schedule('0 7 * * 1-5', async () => {  // Runs at 7 AM, Monday through Friday
        try {
            await fetchMACDData(symbol, 'day', 'close', 12, 26, 9);
            console.log(`Daily MACD data for ${symbol} fetched and processed.`);
        } catch (error) {
            console.error(`Failed to fetch daily MACD data for ${symbol}:`, error);
        }
    }, {
        timezone: "America/Chicago"
    });

    console.log('Scheduled daily MACD updates for SPY at 7 AM CT, Monday through Friday.');
}

module.exports = { scheduleMACDUpdates };