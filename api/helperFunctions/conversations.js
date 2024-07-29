const express = require('express');
const { createConversation, getAllConversations, getConversationById, updateConversation, deleteConversation } = require('../../db/helperFunctions/conversations');

const conversationsRouter = express.Router();

// Create Conversation
conversationsRouter.post('/', async (req, res) => {
    console.log("POST /conversations - Request received");
    const { userId } = req.body;
    console.log("POST /conversations - userId:", userId);

    if (!userId) {
        return res.status(400).json({ message: 'userId is required' });
    }

    try {
        const conversation = await createConversation({ userId });
        console.log("POST /conversations - Conversation created:", conversation);
        res.status(201).json(conversation);
    } catch (error) {
        console.error("POST /conversations - Error creating conversation:", error);
        res.status(500).json({ message: 'Internal server error' });
    }
});

// Get All Conversations
conversationsRouter.get('/', async (req, res) => {
    console.log("GET /conversations - Request received");

    try {
        const conversations = await getAllConversations();
        console.log("GET /conversations - Conversations fetched:", conversations);
        res.status(200).json(conversations);
    } catch (error) {
        console.error("GET /conversations - Error fetching conversations:", error);
        res.status(500).json({ message: 'Internal server error' });
    }
});

// Get Conversation by ID
conversationsRouter.get('/:id', async (req, res) => {
    const { id } = req.params;
    console.log("GET /conversations/:id - Request received for id:", id);

    try {
        const conversation = await getConversationById(id);
        if (conversation) {
            console.log("GET /conversations/:id - Conversation fetched:", conversation);
            res.status(200).json(conversation);
        } else {
            console.log("GET /conversations/:id - Conversation not found for id:", id);
            res.status(404).json({ message: 'Conversation not found' });
        }
    } catch (error) {
        console.error("GET /conversations/:id - Error fetching conversation:", error);
        res.status(500).json({ message: 'Internal server error' });
    }
});

// Update Conversation
conversationsRouter.put('/:id', async (req, res) => {
    const { id } = req.params;
    const fields = req.body;
    console.log("PUT /conversations/:id - Request received for id:", id, "with fields:", fields);

    try {
        const updatedConversation = await updateConversation(id, fields);
        console.log("PUT /conversations/:id - Conversation updated:", updatedConversation);
        res.status(200).json(updatedConversation);
    } catch (error) {
        console.error("PUT /conversations/:id - Error updating conversation:", error);
        res.status(500).json({ message: 'Internal server error' });
    }
});

// Delete Conversation
conversationsRouter.delete('/:id', async (req, res) => {
    const { id } = req.params;
    console.log("DELETE /conversations/:id - Request received for id:", id);

    try {
        const deletedConversation = await deleteConversation(id);
        console.log("DELETE /conversations/:id - Conversation deleted:", deletedConversation);
        res.status(200).json(deletedConversation);
    } catch (error) {
        console.error("DELETE /conversations/:id - Error deleting conversation:", error);
        res.status(500).json({ message: 'Internal server error' });
    }
});

module.exports = conversationsRouter;