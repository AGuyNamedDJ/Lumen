// const axios = require('axios');
// const { storeAggregatesData } = require('../../../db/indicators/aggregates');
// require('dotenv').config();
// const POLYGON_API_KEY = process.env.POLYGON_API_KEY;
// const { polygonApiRequest } = require('../../../utils/polygonApiClient');

// // Function to fetch aggregates data for a specific date from an API
// const fetchAggregatesData = async (symbol, multiplier, date) => {
//     try {
//         const timespan = 'day';  // Standardize to 'day'

//         // Ensure 'date' is correctly formatted as 'YYYY-MM-DD'
//         const formattedDate = new Date(date).toISOString().split('T')[0];

//         console.log(`Fetching Aggregates data for ${symbol} on ${formattedDate}`);

//         const endpoint = `/v2/aggs/ticker/${symbol}/range/${multiplier}/${timespan}/${formattedDate}/${formattedDate}`;
//         const params = {
//             apiKey: POLYGON_API_KEY,
//         };

//         console.log(`Requesting Polygon API with endpoint: ${endpoint} and params: ${JSON.stringify(params)}`);

//         const response = await polygonApiRequest(endpoint, params);

//         console.log('Received response from Polygon API:', JSON.stringify(response, null, 2));

//         if (response && response.results && Array.isArray(response.results)) {
//             const aggregatesData = response.results.map(data => ({
//                 symbol: symbol,
//                 multiplier: multiplier,
//                 timespan: timespan,
//                 timestamp: data.t,  // Use the raw timestamp without conversion
//                 open: data.o,
//                 high: data.h,
//                 low: data.l,
//                 close: data.c,
//                 volume: data.v.toLocaleString(),
//             }));

//             console.log('Formatted Aggregates Data:', JSON.stringify(aggregatesData, null, 2));

//             await storeAggregatesData(aggregatesData);
//             console.log('Aggregates data stored successfully!');
//         } else {
//             console.error('Unexpected response format from Polygon API:', response);
//         }
//     } catch (error) {
//         console.error('Error fetching aggregates data:', error);
//     }
// };

// module.exports = { fetchAggregatesData };