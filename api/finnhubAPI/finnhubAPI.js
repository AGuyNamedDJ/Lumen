const finnhub = require('finnhub');
require('dotenv').config();

// Configure API key authorization: api_key
const apiClient = new finnhub.ApiClient();
const finnhubClient = new finnhub.DefaultApi(apiClient);
apiClient.authentications['api_key'].apiKey = process.env.FINNHUB_API_KEY;

// Function to get real-time stock quotes
function getRealTimeQuotes(symbol, callback) {
    finnhubClient.quote(symbol, (error, data, response) => {
        if (error) {
            console.error('Error fetching real-time quotes:', error);
            return callback(error, null);
        }
        callback(null, data);
    });
}

// New function to get SPX price
async function getSPXPrice() {
    return new Promise((resolve, reject) => {
        getRealTimeQuotes('SPY', (error, data) => {
            if (error) {
                return reject(error);
            }
            resolve(data.c); // 'c' is the current price
        });
    });
}

module.exports = {
    getRealTimeQuotes,
    getSPXPrice
};