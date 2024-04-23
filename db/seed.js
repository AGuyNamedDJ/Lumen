// Import Client & Exports;
const { client } = require('./index');

// File Imports
const importSpecificCSVFile = require('./importHistoricalData');
const { createUser, getAllUsers, getUserById, getUserByUsername, deleteUser, updateUser } = require('./user');

// Methods: Drop Tables
async function dropTables(){
    try {
        console.log("Dropping tables... ");
        await client.query(`
        DROP TABLE IF EXISTS users CASCADE;
        DROP TABLE IF EXISTS strategies CASCADE;
        DROP TABLE IF EXISTS trades CASCADE;
        DROP TABLE IF EXISTS decision_rules;
        DROP TABLE IF EXISTS alerts CASCADE;
        DROP TABLE IF EXISTS market_data CASCADE;
        DROP TABLE IF EXISTS audit_logs CASCADE;
        DROP TABLE IF EXISTS configurations CASCADE;
        DROP TABLE IF EXISTS historical_spx CASCADE;
        DROP TABLE IF EXISTS real_time_spx CASCADE;
    `);
        console.log("Finished dropping tables.")
    } catch(error){
        console.log("Error dropping tables!")
        console.log(error)
    }
}

async function createTables() {
    try {
        console.log('Starting to build tables...');
        await client.query(`
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(255) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            role VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS historical_spx (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            open NUMERIC NOT NULL,
            high NUMERIC NOT NULL,
            low NUMERIC NOT NULL,
            close NUMERIC NOT NULL,
            volume BIGINT
        );
        CREATE TABLE IF NOT EXISTS strategies (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) UNIQUE NOT NULL,
            description TEXT
        );
        CREATE TABLE IF NOT EXISTS trades (
            id SERIAL PRIMARY KEY,
            strategy_id INTEGER REFERENCES strategies(id),
            open_time TIMESTAMP NOT NULL,
            close_time TIMESTAMP,
            status VARCHAR(50) NOT NULL,
            entry_price NUMERIC,
            exit_price NUMERIC,
            profit_loss NUMERIC
        );
        CREATE TABLE IF NOT EXISTS decision_rules (
            id SERIAL PRIMARY KEY,
            strategy_id INTEGER REFERENCES strategies(id),
            parameter_name VARCHAR(255) NOT NULL,
            value TEXT NOT NULL,
            description TEXT
        );
        CREATE TABLE IF NOT EXISTS alerts (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            trade_id INTEGER REFERENCES trades(id),
            strategy_id INTEGER REFERENCES strategies(id),
            message TEXT NOT NULL,
            alert_type VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS market_data (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            open_price NUMERIC NOT NULL,
            close_price NUMERIC NOT NULL,
            high_price NUMERIC NOT NULL,
            low_price NUMERIC NOT NULL,
            volume BIGINT
        );
        CREATE TABLE IF NOT EXISTS audit_logs (
            id SERIAL PRIMARY KEY,
            action_type VARCHAR(255) NOT NULL,
            description TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS configurations (
            id SERIAL PRIMARY KEY,
            key VARCHAR(255) UNIQUE NOT NULL,
            value TEXT NOT NULL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS real_time_spx (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            open NUMERIC NOT NULL,
            high NUMERIC NOT NULL,
            low NUMERIC NOT NULL,
            close NUMERIC NOT NULL,
            volume BIGINT
        );
    `);
    console.log('Finished building tables.');
    } catch (error) {
        console.error('Error building tables!');
        console.log(error);
    }
}
// Create
async function createInitialUsers() {
    console.log("Creating initial users...");
    try {
        await createUser({
            username: 'Admin', 
            password: 'admin1', 
            email: 'admin@gmail.com',
            role: 'admin'
        });

        console.log("Finished creating initial users.");
    } catch (error) {
        console.error("Error creating initial users!");
        console.error(error);
    }
};

// Rebuild DB
async function rebuildDB() {
    try {
        await client.connect();
        await dropTables();
        await createTables();
        await createInitialUsers();
        console.log('Tables have been successfully created.');
    } catch (error) {
        console.error("Error during rebuildDB!", error);
    }
}

// Seed and Import
async function seedAndImport() {
    try {
        await rebuildDB();
        await importSpecificCSVFile(); // Import historical data after rebuilding tables
        console.log('Seed and import completed successfully.');
    } catch (error) {
        console.error("Error during seedAndImport!", error);
    } finally {
        await client.end(); // Close client connection after all operations are done
    }
}

seedAndImport();