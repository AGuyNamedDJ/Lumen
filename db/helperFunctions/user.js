// improts
const { client } = require("../index");
const bcrypt = require('bcrypt');
const validator = require('validator');

// Validation Functions
function validateUsername(username) {
    return typeof username === 'string' && username.trim().length >= 3 && username.trim().length <= 30;
};

function validatePassword(password) {
    return typeof password === 'string' && password.length >= 8;
};

function validateEmail(email) {
    return validator.isEmail(email);
};

function validateName(name) {
    return typeof name === 'string' && name.trim().length > 0;
};

function validatePhoneNumber(phone) {
    return typeof phone === 'string' && validator.isMobilePhone(phone, 'any', { strictMode: false });
};

function validateDateOfBirth(dob) {
    return validator.isDate(dob);
};

// Create User
async function createUser({ username, password, email, first_name, last_name, phone_number, date_of_birth, profile_picture_url }) {
    const SALT_COUNT = 10;

    // Validate input
    console.log(`Validating password: ${password}`);
    if (!validateUsername(username)) {
        throw new Error('Invalid username. Must be between 3 and 30 characters.');
    }
    if (!validatePassword(password)) {
        throw new Error('Invalid password. Must be at least 8 characters long.');
    }
    if (!validateEmail(email)) {
        throw new Error('Invalid email format.');
    }
    if (!validateName(first_name) || !validateName(last_name)) {
        throw new Error('Invalid name. Must not be empty.');
    }
    if (!validatePhoneNumber(phone_number)) {
        throw new Error('Invalid phone number!');
    }
    if (!validateDateOfBirth(date_of_birth)) {
        throw new Error('Invalid date of birth! Must be a valid date.');
    }

    try {
        console.log(`Hashing password for ${username}`);
        const hashedPassword = await bcrypt.hash(password, SALT_COUNT);
        console.log(`Password hashed for ${username}, inserting into database`);

        const result = await client.query(`
            INSERT INTO users(username, password, email, first_name, last_name, phone_number, date_of_birth, profile_picture_url)
            VALUES($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (username) DO NOTHING
            RETURNING id, username, email, first_name, last_name, phone_number, date_of_birth, profile_picture_url, created_at;
        `, [username, hashedPassword, email, first_name, last_name, phone_number, date_of_birth, profile_picture_url]);

        const user = result.rows[0];
        if (user) {
            console.log(`User ${username} inserted into database:`, user);
            return user;
        } else {
            console.log(`User creation failed: Username ${username} already exists.`);
            return null; // Return null if user wasn't created due to username conflict
        }
    } catch (error) {
        console.error(`Could not create user ${username}: ${error}`);
        throw error;
    }
};

// Get All Users
async function getAllUsers() {
    try {
        console.log("Fetching all users from the database.");
        const result = await client.query(`
            SELECT id, username, email, first_name, last_name, phone_number, date_of_birth, created_at
            FROM users;
        `);

        if (result.rows.length > 0) {
            console.log(`Retrieved ${result.rows.length} users from the database.`);
            return result.rows;
        } else {
            console.log("No users found in the database.");
            return [];
        }
    } catch (error) {
        console.error("Error fetching users from the database:", error);
        throw new Error("Error retrieving users from the database!");
    }
};

// getUserById
async function getUserById(id) {
    try {
        console.log(`Fetching user with ID: ${id}`);
        const { rows: [ user ] } = await client.query(`
            SELECT id, username, email, first_name, last_name, phone_number, date_of_birth, created_at
            FROM users
            WHERE id = $1;
        `, [id]);

        if (!user) {
            console.log(`User with ID: ${id} not found!`);
            return null;
        } else {
            console.log(`User with ID: ${id} retrieved successfully.`);
            return user;
        }
    } catch (error) {
        console.error(`Error retrieving user with ID: ${id}`);
        console.error("Error details:", error);
        throw new Error(`Failed to retrieve user due to a server error!`);
    }
};

// getUserByUsername
async function getUserByUsername(username) {
    try {
        console.log(`Fetching user with username: ${username}`);
        const { rows: [user] } = await client.query(`
            SELECT id, username, email, first_name, last_name, phone_number, date_of_birth, created_at
            FROM users
            WHERE username = $1;
        `, [username]);

        if (!user) {
            console.log(`No user found with username: ${username}`);
            return null;
        } else {
            console.log(`User with username: ${username} retrieved successfully:`, user);
            return user;
        }
    } catch (error) {
        console.error(`Error retrieving user with username: ${username}`);
        console.error("Error details:", error);
        throw new Error(`Failed to retrieve user due to a server error!`);
    }
};

// Update User
async function updateUser(username, fields = {}) {
    // Validate input
    if (!username || Object.keys(fields).length === 0) {
        console.log("No updates or username provided.");
        return null; // Early exit if no fields provided or username is missing
    }

    const values = [];
    const setClauses = [];

    Object.keys(fields).forEach((key, index) => {
        if (fields[key] !== undefined) { // Ensure only defined fields are updated
            values.push(fields[key]);
            setClauses.push(`"${key}"=$${index + 1}`);
        }
    });

    // If the password field is present, hash the new password
    if (fields.password) {
        console.log("Hashing new password.");
        values[values.indexOf(fields.password)] = await bcrypt.hash(fields.password, 10);
    }
    
    // Check if there are fields to update
    if (setClauses.length === 0) {
        console.log("No valid fields to update.");
        return null;
    }

    values.push(username);

    try {
        console.log(`Updating user ${username}`);
        const { rows: [user] } = await client.query(`
            UPDATE users
            SET ${setClauses.join(', ')}
            WHERE username=$${values.length}
            RETURNING id, username, email, first_name, last_name, phone_number, date_of_birth, profile_picture_url, created_at, updated_at;
        `, values);

        if (!user) {
            console.log(`User ${username} not found.`);
            return null;
        }

        console.log(`User ${username} updated successfully.`);
        return user;
    } catch (error) {
        console.error(`Could not update user ${username}: ${error}`);
        throw new Error(`Failed to update user due to a server error.`);
    }
};

// Delete User
async function deleteUser(username) {
    try {
        console.log(`Attempting to delete user ${username}`);
        await client.query('BEGIN');

        // Check if the user exists
        const userCheck = await client.query(`
            SELECT id FROM users WHERE username = $1
        `, [username]);

        if (userCheck.rows.length === 0) {
            console.log(`User ${username} not found, cancelling deletion.`);
            await client.query('ROLLBACK');  // Rollback if no user found
            return null;  // Indicate that no user was found
        }

        console.log(`User ${username} found, proceeding to delete.`);
        // Proceed to delete user
        const result = await client.query(`
            DELETE FROM users WHERE username = $1 RETURNING username
        `, [username]);

        await client.query('COMMIT'); 
        console.log(`User ${username} successfully deleted.`);

        return result.rows[0]; 
    } catch (error) {
        await client.query('ROLLBACK');  // Rollback on error
        console.error(`Failed to delete user ${username}: ${error}`);
        throw new Error(`Failed to delete user due to a server error.`);
    }
};

// Login User
async function loginUser({ username, password }) {
    try {
        // Fetch user by username
        const result = await client.query(`
            SELECT * FROM users 
            WHERE username = $1;
        `, [username]);

        // Check if user exists
        const user = result.rows[0];
        if (!user) {
            throw new Error('User not found!');
        }

        // Compare provided password with stored hashed password
        const isPasswordValid = await bcrypt.compare(password, user.password);
        if (!isPasswordValid) {
            throw new Error('Invalid password!');
        }

        // Remove password from user object before returning
        const { password: _, ...userWithoutPassword } = user;

        return userWithoutPassword;
    } catch (error) {
        console.error(`Could not log in user ${username}`);
        console.error(error);
        throw new Error('Failed to log in user due to a server error!');
    }
};

module.exports = {
    createUser,
    getAllUsers,
    getUserById,
    getUserByUsername,
    deleteUser,
    updateUser,
    loginUser
};