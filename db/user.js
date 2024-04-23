const { client } = require('./index');
const bcrypt = require('bcrypt');

// createUser
async function createUser({ username, password, email, role }) {
    try {
        const SALT_COUNT = 10;
        const hashedPassword = await bcrypt.hash(password, SALT_COUNT);

        const result = await client.query(`
            INSERT INTO users(username, password, email, role)
            VALUES($1, $2, $3, $4)
            RETURNING id, username, email, role, created_at;
        `, [username, hashedPassword, email, role]);

        return result.rows[0];
    } catch (error) {
        console.error(`Error creating user: ${error}`);
        throw error;
    }
};

// getAllUsers
async function getAllUsers() {
    try {
        const result = await client.query(`
            SELECT id, username, email, role, created_at
            FROM users;
        `);
        return result.rows;
    } catch (error) {
        console.error(`Error retrieving all users: ${error}`);
        throw error;
    }
};

// getUserById
async function getUserById(id) {
    try {
        const result = await client.query(`
            SELECT id, username, email, role, created_at
            FROM users
            WHERE id = $1;
        `, [id]);

        return result.rows[0];
    } catch (error) {
        console.error(`Error retrieving user by ID: ${error}`);
        throw error;
    }
};

// getUserByUsername
async function getUserByUsername(username) {
    try {
        const result = await client.query(`
            SELECT id, username, email, role, created_at
            FROM users
            WHERE username = $1;
        `, [username]);

        return result.rows[0];
    } catch (error) {
        console.error(`Error retrieving user by username: ${error}`);
        throw error;
    }
};

// deleteUser
async function deleteUser(id) {
    try {
        const result = await client.query(`
            DELETE FROM users
            WHERE id = $1
            RETURNING *;
        `, [id]);

        return result.rows[0];
    } catch (error) {
        console.error(`Error deleting user: ${error}`);
        throw error;
    }
};

// updateUser
async function updateUser(id, fields = {}) {
    const entries = Object.entries(fields);
    const setString = entries.map(([key, _], index) => `${key}=$${index + 2}`).join(", ");
    if (!setString) return;

    if (fields.password) {
        fields.password = await bcrypt.hash(fields.password, 10);
    }

    const values = Object.values(fields);

    try {
        const result = await client.query(`
            UPDATE users
            SET ${setString}
            WHERE id = $1
            RETURNING id, username, email, role, created_at;
        `, [id, ...values]);

        return result.rows[0];
    } catch (error) {
        console.error(`Error updating user: ${error}`);
        throw error;
    }
};

module.exports = {
    createUser,
    getAllUsers,
    getUserById,
    getUserByUsername,
    deleteUser,
    updateUser
};