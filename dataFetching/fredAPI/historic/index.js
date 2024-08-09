const { fetchAverageHourlyEarningsData } = require('./averageHourlyEarningsData');
const { fetchConsumerConfidenceData } = require('./consumerConfidenceData');

async function fetchAllHistoricFredAPIData() {
    await fetchAverageHourlyEarningsData();
    await fetchConsumerConfidenceData();
    // Call other fetch functions for different data points
}

module.exports = { fetchAllHistoricFredAPIData };