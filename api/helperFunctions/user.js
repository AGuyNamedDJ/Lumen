const express = require('express');
const jwt = require('jsonwebtoken');
const { createUser, getAllUsers, getUserById, getUserByUsername, deleteUser, updateUser } = require('../../db/helperFunctions/user');
const JWT_SECRET = process.env.JWT_SECRET;
console.log('JWT_SECRET during token verification:', JWT_SECRET);

const userRouter = express.Router();

userRouter.get('/me', async (req, res) => {
    const authHeader = req.headers.authorization;

    if (!authHeader) {
        console.log('Authorization header is missing');
        return res.status(401).json({ error: 'Authorization header is missing' });
    }

    const token = authHeader.split(' ')[1];

    try {
        console.log('Received token:', token);
        const decoded = jwt.verify(token, JWT_SECRET);
        console.log('Decoded token:', decoded);

        const userId = Number(decoded.id);
        if (isNaN(userId)) {
            console.error('ID in the decoded token is not a number:', decoded.id);
            return res.status(400).json({ error: 'Invalid token payload: ID is not a number' });
        }

        console.log('Extracted user ID:', userId);
        const user = await getUserById(userId);

        if (!user) {
            console.log(`User with ID ${userId} not found`);
            return res.status(404).json({ error: 'User not found' });
        }

        res.status(200).json({ user });
    } catch (error) {
        console.error('Error verifying token or fetching user:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

module.exports = userRouter;