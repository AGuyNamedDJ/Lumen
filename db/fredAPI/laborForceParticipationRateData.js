const { client } = require('../index');

// Store Labor Force Participation data
const storeLaborForceParticipationData = async ({ date, value }) => {
    try {
        console.log(`Storing Labor Force Participation Data: Date - ${date}, Value - ${value}`);
        const query = `
            INSERT INTO labor_force_participation_rate_data (date, value)
            VALUES ($1, $2)
            ON CONFLICT (date) DO NOTHING
            RETURNING *;
        `;
        const values = [date, value];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error storing Labor Force Participation data:', error);
    }
};

// Get all Labor Force Participation data
const getAllLaborForceParticipationData = async () => {
    try {
        console.log('Fetching all Labor Force Participation data');
        const query = 'SELECT * FROM labor_force_participation_rate_data ORDER BY date DESC;';
        const res = await client.query(query);
        return res.rows;
    } catch (error) {
        console.error('Error fetching Labor Force Participation data:', error);
    }
};

// Get Labor Force Participation data by date
const getLaborForceParticipationDataByDate = async (date) => {
    try {
        console.log(`Fetching Labor Force Participation data for date: ${date}`);
        const query = 'SELECT * FROM labor_force_participation_rate_data WHERE date = $1;';
        const values = [date];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error fetching Labor Force Participation data by date:', error);
    }
};

// Update Labor Force Participation data by date
const updateLaborForceParticipationDataByDate = async (date, value) => {
    try {
        console.log(`Updating Labor Force Participation data for date: ${date} with value: ${value}`);
        const query = `
            UPDATE labor_force_participation_rate_data
            SET value = $2, updated_at = CURRENT_TIMESTAMP
            WHERE date = $1
            RETURNING *;
        `;
        const values = [date, value];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error updating Labor Force Participation data by date:', error);
    }
};

// Delete Labor Force Participation data by date
const deleteLaborForceParticipationDataByDate = async (date) => {
    try {
        console.log(`Deleting Labor Force Participation data for date: ${date}`);
        const query = 'DELETE FROM labor_force_participation_rate_data WHERE date = $1 RETURNING *;';
        const values = [date];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error deleting Labor Force Participation data by date:', error);
    }
};

module.exports = {
    storeLaborForceParticipationData,
    getAllLaborForceParticipationData,
    getLaborForceParticipationDataByDate,
    updateLaborForceParticipationDataByDate,
    deleteLaborForceParticipationDataByDate,
};