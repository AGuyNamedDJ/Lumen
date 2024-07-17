const express = require('express');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcrypt');
const { getUserByUsername, loginUser } = require('../../db/helperFunctions/user');
const JWT_SECRET = process.env.JWT_SECRET;

const loginRouter = express.Router();

loginRouter.post('/', async (req, res) => {
    const { username, password } = req.body;

    try {
        console.log('Attempting to log in user with username:', username); // Log username
        const user = await loginUser({ username, password });
        
        if (!user) {
            return res.status(401).json({ error: 'Invalid credentials' });
        }

        // Generate a JWT token
        const token = jwt.sign({ id: user.id, username: user.username }, JWT_SECRET, { expiresIn: '1h' });

        console.log('Generated token:', token); // Log the generated token
        console.log('JWT Secret during token generation:', JWT_SECRET); // Log the secret used

        // Send the token to the client
        res.status(200).json({ token, user: { id: user.id, username: user.username } });
    } catch (error) {
        console.error('Failed to log in:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

module.exports = loginRouter;