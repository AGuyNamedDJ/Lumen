const yahooFinance = require('yahoo-finance2').default;
const { createRealTimeVIXRecord } = require('../../db/helperFunctions/realTimeVIX');
const moment = require('moment-timezone');

const fetchVIXData = async () => {
    try {
        const symbol = '^VIX';

        // Fetch the latest quote for VIX
        const result = await yahooFinance.quote(symbol);

        const timestamp = moment().tz('America/Chicago').format(); // Set timestamp to Central Time
        const current_price = result.regularMarketPrice;
        const volume = result.regularMarketVolume || null;
        const open = result.regularMarketOpen || null;
        const high = result.regularMarketDayHigh || null;
        const low = result.regularMarketDayLow || null;
        const close = result.regularMarketPreviousClose || null;

        // Store the latest VIX data
        await createRealTimeVIXRecord({ timestamp, current_price, volume, conditions: null, open, high, low, close });
        console.log(`Stored VIX Data: Timestamp - ${timestamp}, Price - ${current_price}`);
    } catch (error) {
        console.error('Error fetching VIX data:', error);
    }
};

module.exports = { fetchVIXData };