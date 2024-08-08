const { client } = require('../index');

// Store Nonfarm Payroll Employment data
const storeNonfarmPayrollEmploymentData = async ({ date, value }) => {
    try {
        console.log(`Storing Nonfarm Payroll Employment Data: Date - ${date}, Value - ${value}`);
        const query = `
            INSERT INTO nonfarm_payroll_employment_data (date, value)
            VALUES ($1, $2)
            ON CONFLICT (date) DO NOTHING
            RETURNING *;
        `;
        const values = [date, value];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error storing Nonfarm Payroll Employment data:', error);
    }
};

// Get all Nonfarm Payroll Employment data
const getAllNonfarmPayrollEmploymentData = async () => {
    try {
        console.log('Fetching all Nonfarm Payroll Employment data');
        const query = 'SELECT * FROM nonfarm_payroll_employment_data ORDER BY date DESC;';
        const res = await client.query(query);
        return res.rows;
    } catch (error) {
        console.error('Error fetching Nonfarm Payroll Employment data:', error);
    }
};

// Get Nonfarm Payroll Employment data by date
const getNonfarmPayrollEmploymentDataByDate = async (date) => {
    try {
        console.log(`Fetching Nonfarm Payroll Employment data for date: ${date}`);
        const query = 'SELECT * FROM nonfarm_payroll_employment_data WHERE date = $1;';
        const values = [date];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error fetching Nonfarm Payroll Employment data by date:', error);
    }
};

// Update Nonfarm Payroll Employment data by date
const updateNonfarmPayrollEmploymentDataByDate = async (date, value) => {
    try {
        console.log(`Updating Nonfarm Payroll Employment data for date: ${date} with value: ${value}`);
        const query = `
            UPDATE nonfarm_payroll_employment_data
            SET value = $2, updated_at = CURRENT_TIMESTAMP
            WHERE date = $1
            RETURNING *;
        `;
        const values = [date, value];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error updating Nonfarm Payroll Employment data by date:', error);
    }
};

// Delete Nonfarm Payroll Employment data by date
const deleteNonfarmPayrollEmploymentDataByDate = async (date) => {
    try {
        console.log(`Deleting Nonfarm Payroll Employment data for date: ${date}`);
        const query = 'DELETE FROM nonfarm_payroll_employment_data WHERE date = $1 RETURNING *;';
        const values = [date];
        const res = await client.query(query, values);
        return res.rows[0];
    } catch (error) {
        console.error('Error deleting Nonfarm Payroll Employment data by date:', error);
    }
};

module.exports = {
    storeNonfarmPayrollEmploymentData,
    getAllNonfarmPayrollEmploymentData,
    getNonfarmPayrollEmploymentDataByDate,
    updateNonfarmPayrollEmploymentDataByDate,
    deleteNonfarmPayrollEmploymentDataByDate,
};