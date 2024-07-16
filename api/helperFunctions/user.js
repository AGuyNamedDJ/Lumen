const express = require('express');
const jwt = require('jsonwebtoken');
const { createUser, getAllUsers, getUserById, getUserByUsername, deleteUser, updateUser } = require('../../db/helperFunctions/user');
const JWT_SECRET = process.env.JWT_SECRET;

const userRouter = express.Router();

// Create User
userRouter.post('/', async (req, res) => {
    try {
        const user = await createUser(req.body);
        res.status(201).json({ user });
    } catch (error) {
        console.error('Error creating user:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// Get All Users
userRouter.get('/', async (req, res) => {
    try {
        const users = await getAllUsers();
        res.status(200).json({ users });
    } catch (error) {
        console.error('Error fetching users:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// Get User by ID
userRouter.get('/:id', async (req, res) => {
    try {
        const user = await getUserById(req.params.id);
        if (!user) {
            return res.status(404).json({ error: 'User not found' });
        }
        res.status(200).json({ user });
    } catch (error) {
        console.error('Error fetching user by ID:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// Update User
userRouter.put('/:username', async (req, res) => {
    try {
        const updatedUser = await updateUser(req.params.username, req.body);
        if (!updatedUser) {
            return res.status(404).json({ error: 'User not found' });
        }
        res.status(200).json({ user: updatedUser });
    } catch (error) {
        console.error('Error updating user:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// Delete User
userRouter.delete('/:username', async (req, res) => {
    try {
        const deletedUser = await deleteUser(req.params.username);
        if (!deletedUser) {
            return res.status(404).json({ error: 'User not found' });
        }
        res.status(200).json({ user: deletedUser });
    } catch (error) {
        console.error('Error deleting user:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// Get Current User
userRouter.get('/me', async (req, res) => {
    const authHeader = req.headers.authorization;

    if (!authHeader) {
        return res.status(401).json({ error: 'Authorization header is missing' });
    }

    const token = authHeader.split(' ')[1];

    try {
        const decoded = jwt.verify(token, JWT_SECRET);
        const user = await getUserById(decoded.id);

        if (!user) {
            return res.status(404).json({ error: 'User not found' });
        }

        res.status(200).json({ user });
    } catch (error) {
        console.error('Error verifying token:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

module.exports = userRouter;