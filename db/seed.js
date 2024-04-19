// Import Client & Exports;
const { create } = require('domain');
const { client } = require('./index');

// File Imports

// Methods
async function dropTables(){
    try {
        console.log("Dropping tables... ");
        await client.query(`
        DROP TABLE IF EXISTS market_data CASCADE;
        DROP TABLE IF EXISTS strategies CASCADE;
        DROP TABLE IF EXISTS trades CASCADE;
        DROP TABLE IF EXISTS decision_rules;
        DROP TABLE IF EXISTS alert CASCADE;
        DROP TABLE IF EXISTS users CASCADE;
        DROP TABLE IF EXISTS audit_logs CASCADE;
        DROP TABLE IF EXISTS configurations CASCADE;
    `);
        console.log("Finished dropping tables.")
    } catch(error){
        console.log("Error dropping tables!")
        console.log(error)
    }
};

